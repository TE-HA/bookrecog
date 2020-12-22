import os
all_path = os.path.abspath('..')

f_front = open(all_path+'/bag/photo_front.txt', 'w')
f_back = open(all_path+'/bag/photo_back.txt', 'w')


pth0 = all_path+'/data/front/front_up'
pth1 = all_path+'/data/back/back_up'

photo_name_list_front = []
photo_name_list_back = []
for root, dirs, files in os.walk(pth0, True):
    for file in files:
        # print(file[:-10])
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
for name in photo:
    # f.writelines(name)
    # f.write('\n')
    f_front.writelines(p1 +'/'+ name + '_front.jpg,' + '0')
    f_front.write('\n')
    f_front.writelines(p2 +'/down_'+ name + '_front.jpg,' + '1')
    f_front.write('\n')
    f_back.writelines(p3 +'/'+ name + '_back.jpg,' + '0')
    f_back.write('\n')
    f_back.writelines(p4 +'/down_'+ name + '_back.jpg,' + '1')
    f_back.write('\n')

