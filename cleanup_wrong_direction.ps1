# Chạy file này ở thư mục gốc project:
# powershell -ExecutionPolicy Bypass -File cleanup_wrong_direction.ps1

Write-Host "Cleaning files not related to the final 3-algorithm direction..."

# Không dùng các repo/solver phụ trong dashboard chính
Remove-Item -Recurse -Force "refs\or-tools" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "refs\attention-learn-to-route" -ErrorAction SilentlyContinue

# Không dùng model tự train đơn giản nữa
Remove-Item -Recurse -Force "src\paper_method" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "src\model" -ErrorAction SilentlyContinue
Remove-Item -Force "src\data\create_training_data.py" -ErrorAction SilentlyContinue

# Không dùng OR-Tools trong bảng chính
Remove-Item -Force "src\baselines\ortools_solver.py" -ErrorAction SilentlyContinue

# Xóa cache Python
Get-ChildItem -Path "." -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Xóa result cũ, giữ lại paper_routes.csv nếu đã có
Get-ChildItem -Path "results" -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne "paper_routes.csv" } |
    Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host "Done. Keep refs\pytorch-drl4vrp as paper reproduction reference."
Write-Host "Next: put Colab output into results\paper_routes.csv, then run:"
Write-Host "python -m src.experiments.run_compare"
Write-Host "streamlit run app.py"