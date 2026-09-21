# make_subset.ps1 - Tạo subset 20 ảnh từ dataset
$src_img = "Data\train\images"
$src_lbl = "Data\train\labels"
$dst_train_img = "Data\subset\train\images"
$dst_train_lbl = "Data\subset\train\labels"
$dst_val_img = "Data\subset\val\images"
$dst_val_lbl = "Data\subset\val\labels"

# Tạo folder
New-Item -ItemType Directory -Force -Path $dst_train_img, $dst_train_lbl, $dst_val_img, $dst_val_lbl | Out-Null

# Lấy 20 ảnh random
$files = Get-ChildItem -Path $src_img -Filter "*.jpg" | Get-Random -Count 20

# Chia: 16 train + 4 val
$train_files = $files[0..15]
$val_files = $files[16..19]

foreach ($f in $train_files) {
    Copy-Item $f.FullName -Destination $dst_train_img
    $lbl = Join-Path $src_lbl ($f.BaseName + ".txt")
    if (Test-Path $lbl) {
        Copy-Item $lbl -Destination $dst_train_lbl
    } else {
        Write-Host "WARN: label không có: $($f.BaseName).txt"
    }
}

foreach ($f in $val_files) {
    Copy-Item $f.FullName -Destination $dst_val_img
    $lbl = Join-Path $src_lbl ($f.BaseName + ".txt")
    if (Test-Path $lbl) {
        Copy-Item $lbl -Destination $dst_val_lbl
    }
}

Write-Host ""
Write-Host "DONE!"
Write-Host "Train: $((Get-ChildItem $dst_train_img).Count) ảnh, $((Get-ChildItem $dst_train_lbl).Count) label"
Write-Host "Val:   $((Get-ChildItem $dst_val_img).Count) ảnh, $((Get-ChildItem $dst_val_lbl).Count) label"