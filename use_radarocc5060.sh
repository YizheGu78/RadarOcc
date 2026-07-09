source "$HOME/miniconda3/etc/profile.d/conda.sh"

while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
    conda deactivate
done

conda activate radarocc5060

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

export TORCH_LIB=$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)

export LD_LIBRARY_PATH="$TORCH_LIB:$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include"
export C_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include"

export TORCH_CUDA_ARCH_LIST="12.0"

export PYTHONPATH="$HOME/projects/RadarOcc:$HOME/projects/RadarOcc/third_party/VoxFormer/deform_attn_3d:$PYTHONPATH"
