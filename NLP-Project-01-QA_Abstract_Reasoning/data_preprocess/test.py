
with open('../data_output/train_x.txt', 'r', encoding='utf-8') as f:
    words = []
    for line in f:
        print(line.split(' '))
        line.strip('')


