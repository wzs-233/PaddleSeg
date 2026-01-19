def make_liteseg_list(train_txt_path, val_txt_path):
    with open(train_txt_path, 'r') as f:
        train_list = f.readlines()
    with open(val_txt_path, 'r') as f:
        val_list = f.readlines()
    for i in range(len(train_list)):
        train_list[i] = train_list[i].strip() + ' ' + train_list[i].replace('fg', 'alpha')
    for i in range(len(val_list)):
        val_list[i] = val_list[i].strip() + ' ' + val_list[i].replace('fg', 'alpha')
    with open('train_list.txt', 'w') as f:
        f.writelines(train_list)
    with open('val_list.txt', 'w') as f:
        f.writelines(val_list)
    
    
if __name__ == '__main__':
    make_liteseg_list('/workspace/paddle_paddle/PaddleSeg/data/MaVeCoDD/train.txt', '//workspace/paddle_paddle/PaddleSeg/data/MaVeCoDD/val.txt')