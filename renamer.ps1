$images = Get-ChildItem "training"

$id = 0
foreach ($image in $images) {
    Write-Host $image
    Rename-Item -LiteralPath "training/$($image)" -NewName "gaben$($id)"
    $id++
}