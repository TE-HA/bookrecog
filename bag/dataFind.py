import os
all_path = os.path.abspath('..')



pth0 = all_path+'/data/front/front_up'
pth1 = all_path+'/data/back/back_up'

photo_name_list_front = []
photo_name_list_back = []
for root, dirs, files in os.walk(pth0, True):
    for file in files:
        print(file[:-10])
        photo_name_list_front.append(file[:-10])
        # f.writelines(pth0 +'/'+ file + ',' + '0')
        # f.write('\n')
for root, dirs, files in os.walk(pth1, True):
    for file in files:
        # print(file[:-10])
        photo_name_list_back.append(file[:-9])
        # f.writelines(pth0 +'/'+ file + ',' + '0')
        # f.write('\n')

photo = list(set(photo_name_list_front).intersection(set(photo_name_list_back)))

p1=all_path+'/data/front/front_up'
p2=all_path+'/data/front/front_down'
p3=all_path+'/data/back/back_up'
p4=all_path+'/data/back/back_down'



front_train = open(all_path+'/bag/front_train.txt', 'w')
front_val = open(all_path+'/bag/front_val.txt', 'w')
front_test = open(all_path+'/bag/front_test.txt', 'w')

back_train = open(all_path+'/bag/back_train.txt', 'w')
back_val = open(all_path+'/bag/back_val.txt', 'w')
back_test = open(all_path+'/bag/back_test.txt', 'w')


fi = [front_train, front_val, front_test, back_train, back_val, back_test]

for name in range(photo.__len__()):
    # f.writelines(name)
    # f.write('\n')
    if name<900:
        f_front = front_train
        f_back = back_train
    elif name > 900 and name < 1200:
        f_front = front_val
        f_back = back_val
    else:
        f_front = front_test
        f_back = back_test

    f_front.writelines(p1 +'/'+ photo[name] + '_front.jpg,' + '0')
    f_front.write('\n')
    f_front.writelines(p2 +'/down_'+ photo[name] + '_front.jpg,' + '1')
    f_front.write('\n')
    f_back.writelines(p3 +'/'+ photo[name] + '_back.jpg,' + '0')
    f_back.write('\n')
    f_back.writelines(p4 +'/down_'+ photo[name] + '_back.jpg,' + '1')
    f_back.write('\n')

