// Binding/adaptation only: all occupancy algorithms execute the pinned,
// unmodified OctoMap C++ sources in ../vendor/octomap (BSD-3-Clause).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <octomap/OcTree.h>
#include <cmath>

namespace py = pybind11;
using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;

static octomap::point3d point(const Array& input) {
    if (input.ndim() != 1 || input.shape(0) != 3)
        throw py::value_error("Expected XYZ vector of length 3");
    auto p = input.unchecked<1>();
    if (!std::isfinite(p(0)) || !std::isfinite(p(1)) || !std::isfinite(p(2)))
        throw py::value_error("XYZ must be finite");
    return octomap::point3d(p(0), p(1), p(2));
}

static octomap::OcTreeKey key(const py::sequence& input) {
    if (py::len(input) != 3) throw py::value_error("Expected three key components");
    octomap::OcTreeKey result;
    for (unsigned axis = 0; axis < 3; ++axis) {
        long k = py::cast<long>(input[axis]);
        if (k < 0 || k >= 65536) throw py::value_error("Key outside depth-16 range");
        result[axis] = static_cast<octomap::key_type>(k);
    }
    return result;
}

static py::tuple key_tuple(const octomap::OcTreeKey& k) {
    return py::make_tuple(k[0], k[1], k[2]);
}

// Queries return probability snapshots, not raw pointers: upstream updates can
// invalidate a node even while the tree itself is still alive (pruning).
struct NodeValue {
    float value;
    explicit NodeValue(float v) : value(v) {}
    double occupancy() const {
        octomap::OcTreeNode node;
        node.setLogOdds(value);
        return node.getOccupancy();
    }
};

static py::object snapshot(const octomap::OcTreeNode* node) {
    if (!node) return py::none();
    return py::cast(NodeValue(node->getLogOdds()));
}

static void valid_probability(double p) {
    if (!std::isfinite(p) || !(p > 0 && p < 1))
        throw py::value_error("Probability must be finite and in (0, 1)");
}

PYBIND11_MODULE(_native, m) {
    m.doc() = "Thin bindings to the unmodified pinned OctoMap OcTree";
    m.attr("source_fingerprint") = OCTOMAP_BINDING_FINGERPRINT;
    m.attr("upstream_commit") = "21a8871d7bbd0aa13c73bbdb4e128821d8279af4";
    py::class_<NodeValue>(m, "NodeValue")
        .def_readonly("value", &NodeValue::value)
        .def("getLogOdds", [](const NodeValue& n) { return n.value; })
        .def("getOccupancy", &NodeValue::occupancy);
    py::class_<octomap::OcTree>(m, "OcTree")
        .def(py::init([](double resolution) {
            if (!std::isfinite(resolution) || resolution <= 0)
                throw py::value_error("Resolution must be finite and positive");
            return std::unique_ptr<octomap::OcTree>(new octomap::OcTree(resolution));
        }))
        .def("setProbHit", [](octomap::OcTree& t, double p) {
            valid_probability(p); if (p < .5) throw py::value_error("p_hit must be >= .5");
            t.setProbHit(p);
        })
        .def("setProbMiss", [](octomap::OcTree& t, double p) {
            valid_probability(p); if (p > .5) throw py::value_error("p_miss must be <= .5");
            t.setProbMiss(p);
        })
        .def("setOccupancyThres", [](octomap::OcTree& t, double p) {
            valid_probability(p); t.setOccupancyThres(p);
        })
        .def("setClampingThresMin", [](octomap::OcTree& t, double p) {
            valid_probability(p); t.setClampingThresMin(p);
        })
        .def("setClampingThresMax", [](octomap::OcTree& t, double p) {
            valid_probability(p); t.setClampingThresMax(p);
        })
        .def("size", &octomap::OcTree::size)
        .def("clear", &octomap::OcTree::clear)
        .def("prune", &octomap::OcTree::prune)
        .def("updateInnerOccupancy", &octomap::OcTree::updateInnerOccupancy)
        .def_property_readonly("root", [](const octomap::OcTree& t) { return snapshot(t.getRoot()); })
        .def("coordToKeyChecked", [](const octomap::OcTree& t, const Array& input) -> py::object {
            octomap::OcTreeKey k;
            if (!t.coordToKeyChecked(point(input), k)) return py::none();
            return key_tuple(k);
        })
        .def("computeRayKeys", [](const octomap::OcTree& t, const Array& a, const Array& b) -> py::object {
            octomap::KeyRay ray;
            if (!t.computeRayKeys(point(a), point(b), ray)) return py::none();
            py::list result;
            for (const auto& k : ray) result.append(key_tuple(k));
            return result;
        })
        .def("setBBXMin", [](octomap::OcTree& t, const Array& p) {
            auto value = point(p); octomap::OcTreeKey k;
            if (!t.coordToKeyChecked(value, k)) throw py::value_error("BBX outside key range");
            t.setBBXMin(value);
        })
        .def("setBBXMax", [](octomap::OcTree& t, const Array& p) {
            auto value = point(p); octomap::OcTreeKey k;
            if (!t.coordToKeyChecked(value, k)) throw py::value_error("BBX outside key range");
            t.setBBXMax(value);
        })
        .def("useBBXLimit", &octomap::OcTree::useBBXLimit)
        .def("insertPointCloud", [](octomap::OcTree& t, const Array& points,
                                    const Array& sensor_origin, double maxrange, bool lazy, bool discrete) {
            if (points.ndim() != 2 || points.shape(1) != 3)
                throw py::value_error("Point cloud must have shape [N,3]");
            if (!std::isfinite(maxrange)) throw py::value_error("maxrange must be finite");
            auto origin = point(sensor_origin);
            auto values = points.unchecked<2>();
            octomap::Pointcloud scan;
            scan.reserve(static_cast<size_t>(points.shape(0)));
            for (py::ssize_t i = 0; i < points.shape(0); ++i) {
                if (!std::isfinite(values(i, 0)) || !std::isfinite(values(i, 1)) || !std::isfinite(values(i, 2)))
                    throw py::value_error("Point cloud must be finite");
                scan.push_back(values(i, 0), values(i, 1), values(i, 2));
            }
            // One complete original scan per call. No per-ray updates, new
            // filtering, parallel insertion, alternative ISM or fusion here.
            // Keep the GIL: avoid concurrent mutation of this same tree.
            t.insertPointCloud(scan, origin, maxrange, lazy, discrete);
        }, py::arg("scan"), py::arg("sensor_origin"), py::arg("maxrange") = -1.,
           py::arg("lazy_eval") = false, py::arg("discretize") = false)
        .def("updateNode", [](octomap::OcTree& t, const py::sequence& k, const py::object& update, bool lazy) {
            auto cell = key(k);
            if (py::isinstance<py::bool_>(update))
                return snapshot(t.updateNode(cell, update.cast<bool>(), lazy));
            float delta = update.cast<float>();
            if (!std::isfinite(delta)) throw py::value_error("log odds update must be finite");
            return snapshot(t.updateNode(cell, delta, lazy));
        }, py::arg("key"), py::arg("occupied_or_log_odds"), py::arg("lazy_eval") = false)
        .def("search", [](const octomap::OcTree& t, const py::sequence& k, unsigned depth) {
            if (depth > 16) throw py::value_error("Depth must be in [0,16]");
            return snapshot(t.search(key(k), depth));
        }, py::arg("key"), py::arg("depth") = 0)
        .def("isNodeOccupied", [](const octomap::OcTree& t, const NodeValue& value) {
            octomap::OcTreeNode node; node.setLogOdds(value.value);
            return t.isNodeOccupied(node);
        })
        .def("iter_leaves", [](const octomap::OcTree& t) {
            py::list result;
            for (auto it = t.begin_leafs(); it != t.end_leafs(); ++it) {
                auto k = it.getKey();
                unsigned span = 1u << (16 - it.getDepth());
                auto lower = py::make_tuple(k[0] & ~(span - 1), k[1] & ~(span - 1), k[2] & ~(span - 1));
                result.append(py::make_tuple(lower, span, NodeValue(it->getLogOdds())));
            }
            return result;
        });
}
