#!/home/rthedin/.conda/envs/ssrs_env_scratch/bin/python

# ----------------------------------------------------------------------- #
# approximateTimeDirectory.py                                             #
#                                                                         #
# Approximate time directories.                                           #
#                                                                         #
# Usage:                                                                  #
# approximateTimeDirectory.py <precision> [<tolerance>]                   #
# where precision is the number of decimal digits to accomodate the time  #
# step used, usually 2 or 3. Tolerance is optional, and defaults to 1e-5  #
#                                                                         #
# Example call:                                                           #
#    cd postProcessing/Low                                                #
#    approximateTimeDirectory.py 2                                        #
#    approximateTimeDirectory.py 0 0.1                                    #
# The last example above refers to a case where you have already rounded  #
# the values, however instead of getting, for example, 100, 200, and 300, #
# you got 100, 199.9, and 299.9, due to the way the sampling directories  #
# were saved by OpenFOAM. So you want to run it again and give precision  #
# 0 (meaning integers), but now the tolerance for approximating 199.9 to  #
# 200 needs to be changed. The optional third argument is the tolerance.  #
#                                                                         #
# Regis Thedin                                                            #
# Aug 22, 2022                                                            #
# regis.thedin@nrel.gov                                                   #
# ----------------------------------------------------------------------- #
import os, sys
import numpy as np

if 4 <= len(sys.argv) <= 1:
    sys.exit('USAGE: '+sys.argv[0]+' <precision> [<tolerance>]')

print(f'Using python {sys.version}')

# Get tolerance
if len(sys.argv) == 3:
    tol = float(sys.argv[2])
    print(f'Using user-specified tolerance {tol}')
else:
    tol = 1e-5

# Get precision
precision = float(sys.argv[1])
if precision%1 != 0:
    raise ValueError(f'precision value must be an integer')
else:
    precision = int(precision)

# Print messages and get answer from user
path = os.getcwd()
print(f'Approximating values with precision {precision} within the following path:')
print(f'{path}')
answer = input(f'Continue? ([y]/n) ').lower()
if answer=='':
    answer='y'
if answer != 'y':
    sys.exit()


# Obtain list of times dirs in str and float
timesStr = os.listdir(path)
timesFlo = [float(t) for t in timesStr]  # clipped at precision 12

# Reorder them
timesFloSort = np.sort(timesFlo)
sortInd =  np.argsort(timesFlo)
timesStrSort = np.array(timesStr)[sortInd]

# Round values up to precision
timesFloSortRound = np.round(timesFloSort,precision)

# Convert to string and strip `.0` if integer
timesStrSortRound = [str(int(t)) if t%1==0 else str(t) for t in timesFloSortRound]


if len(timesStr) != len(timesStrSortRound):
    raise ValueError('Lengths of directory arrays are different. Stopping.')

for i,t in enumerate(timesStrSort):
    if timesStrSort[i]==timesStrSortRound[i]:
        print(f'Time {timesStrSort[i]} does not need to be adjusted. Skipping it.')
    else:
        print(f'Adjusting Time {timesStrSort[i]} to {timesStrSortRound[i]}')
        if abs(float(timesStrSort[i])-float(timesStrSortRound[i])) > tol:
            raise ValueError(f'Trying to approximate {timesStrSort[i]} to {timesStrSortRound[i]}. '\
                             f'These numbers do not seem close enough. Maybe increase the '\
                             f'tolerance (approximateTimeDirectory.py 0 0.1). Stopping now.')
        os.rename(os.path.join(path,timesStrSort[i]), os.path.join(path,timesStrSortRound[i]))

    

