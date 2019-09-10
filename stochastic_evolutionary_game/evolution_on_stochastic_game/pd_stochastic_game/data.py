import pylab

def readfile(file_name):
    f = open(file_name)
    data = f.readlines()
    f.close()
    return data

data = readfile('pri_coop_time.txt')

coops = []
coop1 = []
coop2 = []
for item in data[0][:-1].split(','):
    coops.append(float(item))
for item in data[1][:-1].split(','):
    coop1.append(float(item))
for item in data[2][:-1].split(','):
    coop2.append(float(item))
pylab.figure()
pylab.title('Cooperation Rate with Time')
pylab.xlabel('Time')
pylab.ylabel('Cooperation Fraction')
pylab.plot(coops, 'k')
pylab.plot(coop1, 'r')
pylab.plot(coop2, 'g')
pylab.show()