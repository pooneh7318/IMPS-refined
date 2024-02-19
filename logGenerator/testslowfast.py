import os
from datetime import datetime, timedelta
import random
import pytz
path = 'imps/logs'
now = datetime.now(pytz.timezone('Canada/Eastern'))
#inittime = now.strftime("%H:%M:%S.%f")
rsec = random.randint(0,1)
notelist = [0.47999998927116394, 0.49000000953674316, 0.5, 0.5099999904632568, 
                  0.5199999809265137, 0.5299999713897705, 0.5400000214576721, 0.550000011920929,
                  0.5600000023841858, 0.5699999928474426, 0.5799999833106995, 
                  0.5899999737739563, 0.6000000238418579, 0.6100000143051147,
                  0.6200000047683716, 0.6299999952316284, 0.6399999856948853]
seclist = [0, 1, 2]

index2 = 0
index7 = 0
index8 = 0
index9 = 0
index15 = 0

for y in range(500):
    name = now.strftime("%Y-%m-%dT%H-%M-%S-2d-mdrnn.log")
    completeName = os.path.join(path, name)
    with open(completeName, 'w') as f:
        f.write(now.strftime("%Y-%m-%dT%H:%M:%S.%f,"))
        

        index = random.randint(0,16)

        if index == 2:
            index2 += 1
        if index == 7:
            index7 += 1
        if index == 8:
            index8 += 1
        if index == 15:
            index15 += 1
        if index == 9:
            index9 += 1

        

        
        rnote = notelist[index]
        f.write('interface,')
        f.write(str(rnote))
        f.write('\n')
        for x in range(23):
            rsec = [0]
            # rsec = random.choices(seclist,weights=(65,32,3), k=1)
            # rsec = random.randint(0,1)
            rmicsec = random.randint(0,400000)
            now = now + timedelta(seconds=rsec[0],microseconds=rmicsec)
            f.write(now.strftime("%Y-%m-%dT%H:%M:%S.%f,"))

            # if(index < 16):
            #     index = index + 1
            # else:
            #     index = 0
            index = random.randint(0,16)
            if index == 2:
                index2 += 1
            if index == 7:
                index7 += 1
            if index == 8:
                index8 += 1
            if index == 15:
                index15 += 1
            if index == 9:
                index9 += 1
            rnote = notelist[index]
            f.write('interface,')
            f.write(str(rnote))
            f.write('\n')
        
        for x in range(23):
            rsec = [random.randint(1,2)]
            # rsec = random.choices(seclist,weights=(65,32,3), k=1)
            # rsec = random.randint(0,1)
            rmicsec = random.randint(0,800000)
            now = now + timedelta(seconds=rsec[0],microseconds=rmicsec)
            f.write(now.strftime("%Y-%m-%dT%H:%M:%S.%f,"))

            # if(index < 16):
            #     index = index + 1
            # else:
            #     index = 0
            index = random.randint(0,16)
            if index == 2:
                index2 += 1
            if index == 7:
                index7 += 1
            if index == 8:
                index8 += 1
            if index == 15:
                index15 += 1
            if index == 9:
                index9 += 1
            rnote = notelist[index]
            f.write('interface,')
            f.write(str(rnote))
            f.write('\n')
        
        for x in range(23):
            rsec = [0]
            # rsec = random.choices(seclist,weights=(65,32,3), k=1)
            # rsec = random.randint(0,1)
            rmicsec = random.randint(0,400000)
            now = now + timedelta(seconds=rsec[0],microseconds=rmicsec)
            f.write(now.strftime("%Y-%m-%dT%H:%M:%S.%f,"))

            # if(index < 16):
            #     index = index + 1
            # else:
            #     index = 0
            index = random.randint(0,16)
            if index == 2:
                index2 += 1
            if index == 7:
                index7 += 1
            if index == 8:
                index8 += 1
            if index == 15:
                index15 += 1
            if index == 9:
                index9 += 1
            rnote = notelist[index]
            f.write('interface,')
            f.write(str(rnote))
            f.write('\n')

        for x in range(21):
            rsec = [random.randint(1,2)]
            # rsec = random.choices(seclist,weights=(65,32,3), k=1)
            # rsec = random.randint(0,1)
            rmicsec = random.randint(0,800000)
            now = now + timedelta(seconds=rsec[0],microseconds=rmicsec)
            f.write(now.strftime("%Y-%m-%dT%H:%M:%S.%f,"))

            # if(index < 16):
            #     index = index + 1
            # else:
            #     index = 0
            index = random.randint(0,16)
            if index == 2:
                index2 += 1
            if index == 7:
                index7 += 1
            if index == 8:
                index8 += 1
            if index == 15:
                index15 += 1
            if index == 9:
                index9 += 1
            rnote = notelist[index]
            f.write('interface,')
            f.write(str(rnote))
            f.write('\n')
        

    #f.write('2022-11-02T')
    #f.write(',')

print("index2", index2)
print("index7", index7)
print("index8", index8)
print("index15", index15)
print("index9", index9)
