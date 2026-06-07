# Hướng đúng của project CVRP AGIH

## Ba thuật toán chính

1. **Clarke-Wright**
   - Thuật toán cũ / heuristic truyền thống.
   - Chạy trực tiếp trong project.

2. **Paper-RL-Attention**
   - Thuật toán bài báo.
   - Route được sinh từ repo bài báo/reproduction trên Google Colab.
   - Kết quả Colab cần xuất về `results/paper_routes.csv`.

3. **Proposed-AGIH-2opt**
   - Thuật toán đề xuất.
   - Lấy route của Paper-RL-Attention làm nghiệm khởi tạo.
   - Cải thiện bằng 2-opt và relocate local search.

## File paper_routes.csv cần có format

```csv
Instance,Paper_Distance,Paper_Vehicles,Paper_Runtime,Paper_Routes
CVRP20_0000,5.1234,4,0.032,"[[0,1,5,0],[0,2,3,4,0]]"
```

Trong đó:
- `Instance` phải khớp id trong `data/cvrp20_test.pkl`
- `Paper_Routes` phải là JSON list của các route
- mỗi route bắt đầu/kết thúc bằng depot `0`

## Chạy project

```powershell
python -m src.experiments.run_compare
streamlit run app.py
```

## Các file không còn dùng

- `src/paper_method/*`: model attention tự train đơn giản, không phải thuật toán bài báo gốc.
- `src/model/*`: phục vụ model tự train cũ.
- `src/baselines/ortools_solver.py`: OR-Tools không thuộc 3 thuật toán chính.
- `refs/or-tools`, `refs/attention-learn-to-route`: không cần cho hướng cuối.