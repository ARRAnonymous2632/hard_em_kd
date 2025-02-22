import fileinput
for line in fileinput.input():
    print(line.replace("@@ ", ""), end="")