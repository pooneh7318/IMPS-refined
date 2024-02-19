import os
from datetime import datetime, timedelta
import random
import pytz
import json
path_json = '/Users/pooneh/Documents/project/machine_improviser_data/gesture-rating-experiment/gestures'
path_logs = '/Users/pooneh/imps/logs'
count = 0
i = 0
# MAX = 59965.554455295205
# MIN = -2.619709253311157
# denom = MAX - MIN 
now = datetime.now(pytz.timezone('Canada/Eastern'))
# maxmin = []
# for json_filename in sorted(os.listdir(path_json)):
#     i = 0
#     if json_filename.endswith(".json"):
#         with open(os.path.join(path_json, json_filename), 'r') as read_file:
#             data = json.load(read_file)
#             for i in data['analysis']['spectral']:
#                 maxmin.append((max(data['analysis']['spectral'][i])))
#                 maxmin.append((min(data['analysis']['spectral'][i])))

# rsec = random.randint(0,1)
for json_filename in sorted(os.listdir(path_json)):
    i = 0
    if json_filename.endswith(".json"):
        with open(os.path.join(path_json, json_filename), 'r') as read_file:
            data = json.load(read_file)
            name = now.strftime("%Y-%m-%dT%H-%M-%S-22d-mdrnn.log")
            completeName = os.path.join(path_logs, name)
            with open(completeName, 'w') as f:
                f.write(now.strftime("%Y-%m-%dT%H:%M:%S.%f,"))
                # index = random.randint(0,16)
                # rnote = notelist[index]
                f.write('interface,')
                for c in data['analysis']['spectral']:
                    numerator = data['analysis']['spectral'][c][0] - min(data['analysis']['spectral'][c])
                    norm = numerator / (max(data['analysis']['spectral'][c]) - min(data['analysis']['spectral'][c]))
                    if(c != 'SpectralCentroid' and c!='time'):
                        f.write(str(norm))
                        f.write(',')
                    elif(c == 'SpectralCentroid'):
                        f.write(str(norm))
                f.write('\n')
                for t in data['analysis']['spectral']['time']:
                    msec = t
                    newnow = now + timedelta(seconds=0,milliseconds=msec)
                    f.write(newnow.strftime("%Y-%m-%dT%H:%M:%S.%f,"))
                    f.write('interface,')
                    for y in data['analysis']['spectral']:
                        numerator = data['analysis']['spectral'][y][i] - min(data['analysis']['spectral'][y])
                        norm = numerator / (max(data['analysis']['spectral'][y]) - min(data['analysis']['spectral'][y]))
                    # rsec = random.choices(seclist,weights=(65,32,3), k=1)
                    # rsec = random.randint(0,1)
                        if(y != 'SpectralCentroid' and y!='time'):
                            f.write(str(norm))
                            f.write(',')
                        elif(y == 'SpectralCentroid'):
                            f.write(str(norm))
                    f.write('\n')
                    i += 1
                now = newnow



#inittime = now.strftime("%H:%M:%S.%f")
# # notelist = [0.47999998927116394, 0.49000000953674316, 0.5, 0.5099999904632568, 
#                   0.5199999809265137, 0.5299999713897705, 0.5400000214576721, 0.550000011920929,
#                   0.5600000023841858, 0.5699999928474426, 0.5799999833106995, 
#                   0.5899999737739563, 0.6000000238418579, 0.6100000143051147,
#                   0.6200000047683716, 0.6299999952316284, 0.6399999856948853]
# seclist = [0, 1, 2]


        

    #f.write('2022-11-02T')
    #f.write(',')

