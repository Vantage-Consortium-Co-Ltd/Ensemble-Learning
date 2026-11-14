# Ensemble Learning: Bagging vs Boosting

เทียบ Bagging กับ Boosting บนชุดข้อมูล Iris (Hugging Face `antoniopelusi/iris`) โดยเขียนตัว ensemble เอง
ส่วนโมเดลพื้นฐาน (Decision Tree) ใช้ของ scikit-learn

| | Bagging | Boosting (AdaBoost SAMME) |
|---|---|---|
| วิธีเทรน | ต้นไม้แต่ละต้นเทรน **พร้อมกัน** บนข้อมูลที่สุ่มแบบใส่คืน (bootstrap) | เทรน **ทีละต้น** ต้นถัดไปเน้นตัวอย่างที่ต้นก่อนทายผิด |
| โมเดลพื้นฐาน | ต้นไม้ลึกเต็มที่ (variance สูง) | ต้นไม้ชั้นเดียว / decision stump (bias สูง) |
| ช่วยลด | variance | bias |
| การโหวต | เสียงละ 1 | ถ่วงน้ำหนักด้วย alpha ของแต่ละต้น |

## ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `common.py` | โหลด Iris + แบ่ง train/test 80/20, ฟังก์ชันโหวต, ฟังก์ชันวัดผล |
| `bagging.py` | คลาส `Bagging` เทรนขนานด้วย `ThreadPoolExecutor` |
| `boosting.py` | คลาส `AdaBoost` (SAMME) เขียนด้วย NumPy |
| `visualize.py` | วาดกราฟเทียบความแม่นและเวลา |

## วิธีรัน

```bash
pip install -r requirements.txt
python bagging.py            # ตัวเลือก: --n 50 --workers 4 --scale 1
python boosting.py           # ตัวเลือก: --n 50
python visualize.py          # ได้ output/compare.png
```

## ผลลัพธ์ (test set 30 ตัวอย่าง, 50 ต้น)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Decision Tree (single) | 0.933 | 0.933 |
| Bagging | 0.933 | 0.933 |
| Decision Stump (single) | 0.667 | 0.556 |
| AdaBoost | 0.933 | 0.933 |

- **Boosting** เห็นผลชัด: stump ต้นเดียวแยกได้แค่ 2 จาก 3 พันธุ์ (0.667) แต่พอรวม 50 ต้นได้ 0.933
- **Bagging** บน Iris ไม่ได้แม่นขึ้น เพราะข้อมูลแยกง่ายและต้นไม้ต้นเดียวก็ทำได้ดีอยู่แล้ว
- **ความเร็วของ parallel**: Iris มีแค่ 120 แถว เวลาที่ใช้เปิด thread จึงพอ ๆ กับเวลาที่ประหยัดได้
  ถ้าขยายข้อมูลด้วย `--scale 500` (60,000 แถว) แบบ parallel เทรนเร็วกว่า sequential ประมาณ 4 เท่า (0.18 วินาที เทียบกับ 0.74 วินาที)
  และ accuracy ขึ้นเป็น 0.967

ตัวเลขเวลาเปลี่ยนตามเครื่อง

## หมายเหตุ

- ใช้ thread แทน process เพราะ scikit-learn ปล่อย GIL ระหว่างเทรนต้นไม้ และบน Windows การเปิด process ใหม่ใช้เวลานาน
- AdaBoost ส่ง `sample_weight` ให้ต้นไม้โดยตรง ไม่ต้องสุ่มข้อมูลใหม่ตามน้ำหนัก
- ถ้าต้นไหนทายผิดมากกว่าการเดาสุ่ม (`err >= 1 - 1/K`) จะหยุดเพิ่มต้น
