$images = Get-ChildItem "imgs"

$id = 0
foreach ($image in $images) {
    Write-Host $image
    Rename-Item -LiteralPath "imgs/$($image)" -NewName "gaben$($id)"
    $id++
}