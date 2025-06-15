import zipfile

zip_path = "./trec07p.tgz"
to = "."

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(to)