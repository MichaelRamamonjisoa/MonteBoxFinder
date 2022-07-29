import os
import shutil


print("------- Installing JSONCPP --------")
os.system("git clone https://github.com/open-source-parsers/jsoncpp jsoncpp")

os.chdir("jsoncpp")

os.system("python amalgamate.py")
if os.path.exists("../include/json"):
    shutil.rmtree("../include/json")
shutil.copytree("dist/json", "../include/json")
shutil.copyfile("dist/jsoncpp.cpp", "../src/jsoncpp.cpp")

print("------- JSON install DONE ---------")


print("------- Building project ----------")
os.chdir("..")
if os.path.exists("build"):
    shutil.rmtree("build")
os.mkdir("build")
os.chdir("build")
os.system("cmake -DCMAKE_PREFIX_PATH=../CGAL-5.2.2 -DCMAKE_BUILD_TYPE=Release ..")
os.system("make")
print("------- Project build DONE --------")
