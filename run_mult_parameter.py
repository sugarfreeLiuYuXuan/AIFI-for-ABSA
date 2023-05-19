import os
import subprocess
import sys
from multiprocessing import  Process
import torch

def run(command):
    subprocess.call(command, shell=True)
def sh(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    parameter, result, time_cost = '', '', ''
    for line in iter(p.stdout.readline, b''):
        line = line.rstrip().decode('utf8')
        if 'args:' in line:
            parameter = line
        if 'final best performance:' in line:
            result = line
        if 'Experiment cost:' in line:
            time_cost = line
    return parameter, result, time_cost

def process_1():
    for lr in [1e-5]:
        for shared_weight in [1.0]:
                path=os.path.join('out/Final-Laptop', "Cos-%s-%s"%(lr,shared_weight,))
                cmd = 'python -m absa.run_joint_span ' +' '+ \
                     ' --num_train_epochs ' + str(10) + ' '+ \
                     ' --train_file  laptop14_train.txt ' +' '+ \
                     ' --predict_file  laptop14_test.txt' + ' ' + \
                      ' --output_dir '  + str(path) + ' '+ \
                      ' --learning_rate '  + str(lr) + ' '+ \
                      ' --weight_kl '  + str(0) + ' '+ \
                      ' --shared_weight '  + str(shared_weight) + ' '
                run(cmd)
                sys.stdout.flush()


def process_2():
    for weight_kl in [0.15]:
        for shared_weight in [0.6]:
            path = os.path.join('out/Final-Restaurant', "s-50-%s-%s" % (weight_kl,shared_weight))
            cmd = 'python -m absa.run_joint_span ' + ' ' + \
                  ' --num_train_epochs ' + str(50) + ' ' + \
                  ' --train_file  rest_total_train.txt ' + ' ' + \
                  ' --predict_file  rest_total_test.txt' + ' ' + \
                  ' --output_dir ' + str(path) + ' ' + \
                  ' --learning_rate ' + str(2e-5) + ' ' + \
                  ' --weight_kl ' + str(weight_kl) + ' ' + \
                  ' --shared_weight ' + str(shared_weight) + ' '
            run(cmd)
            sys.stdout.flush()
def process_3():
    for weight_kl in [0]:
        for shared_weight in [1]:
            path = os.path.join('out/Final-Restaurant', "50-%s-%s" % (weight_kl,shared_weight))
            cmd = 'python -m absa.run_joint_span ' + ' ' + \
                  ' --num_train_epochs ' + str(50) + ' ' + \
                  ' --train_file  rest_total_train.txt ' + ' ' + \
                  ' --predict_file  rest_total_test.txt' + ' ' + \
                  ' --output_dir ' + str(path) + ' ' + \
                  ' --learning_rate ' + str(2e-5) + ' ' + \
                  ' --weight_kl ' + str(weight_kl) + ' ' + \
                  ' --shared_weight ' + str(shared_weight) + ' '
            run(cmd)
            sys.stdout.flush()
def process_4():   #2e-5
    for lr in [3e-5,1e-5]:
        for shared_weight in [1]:
              for number in [1,2,3,4,5,6,7,8,9,10]:
                    train_path = os.path.join('twitter%s_train.txt' % (number))
                    test_path = os.path.join('twitter%s_test.txt' % (number))
                    path = os.path.join('out/Twitter', "%s-%s" % (number,lr))
                    cmd = 'python -m absa.run_joint_span ' + ' ' + \
                          ' --num_train_epochs ' + str(50) + ' ' + \
                          ' --train_file  ' +str(train_path)+ ' ' + \
                          ' --predict_file  ' + str(test_path)+' ' + \
                          ' --output_dir ' + str(path) + ' ' + \
                          ' --learning_rate ' + str(lr) + ' ' + \
                          ' --weight_kl ' + str(0) + ' ' + \
                          ' --shared_weight ' + str(shared_weight) + ' '
                    run(cmd)
                    sys.stdout.flush()


if __name__ == '__main__':
    process_1()
    #process_3()
    #process_4()





