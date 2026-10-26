from pathlib import Path
from collections import defaultdict

path = Path(
    "/home/user1/projects/RadarOcc/data/K-Radar-RGB/K-Radar/"
    "K-Radar-official-gt/kradar_revised_label_v2_0/"
    "KRadar_refined_label_by_UWIPL/11/00262_00229.txt"
)

objects = defaultdict(list)

print(f"原始标注：{path}\n")

for line_number, line in enumerate(path.read_text().splitlines(), start=1):
    print(f"{line_number:3d}: {line}")

    fields = [field.strip() for field in line.split(",")]
    if len(fields) == 10 and fields[0] == "*":
        object_id = fields[1]
        objects[object_id].append((line_number, line))

print("\n重复 ID：")
for object_id, rows in objects.items():
    if len(rows) > 1:
        print(f"\nID={object_id}，出现 {len(rows)} 次")
        for line_number, line in rows:
            print(f"  第 {line_number} 行：{line}")