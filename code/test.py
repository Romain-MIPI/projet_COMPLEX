l = [[], []]
i = 0
while (sum([len(l[i]) for i in range(len(l))]) != 0):
    i += 1
    if i == 5:
        break

print(i)