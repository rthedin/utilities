import pandas as pd
import subprocess

# get queue and format it
command = "squeue -o '%.15u %.8a %.15P %3j %.2t %.10M %.5D %.4C  %4r' -S V"
result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
data = [line.split() for line in result.stdout.split('\n') if line.strip()]
df = pd.DataFrame(data[1:], columns=data[0])
df['NODES'] = pd.to_numeric(df['NODES'], errors='coerce')

# split running and queued jobs
running = df[df['ST'] == 'R']
queued  = df[df['ST'] == 'PD']
singlen_running = running[running['NODES'] == 1]
singlen_queued  = queued[queued['NODES'] == 1]
multiplen_running = running[running['NODES'] != 1]
multiplen_queued  = queued[queued['NODES'] != 1]
user_counts_running = running['USER'].value_counts()
user_counts_queued  = queued['USER'].value_counts()
user_counts_running_singlenode   = singlen_running['USER'].value_counts()
user_counts_running_multiplenode = multiplen_running['USER'].value_counts()
user_counts_list_running = [(user, count) for user, count in zip(user_counts_running.index, user_counts_running)]
user_counts_list_queued  = [(user, count) for user, count in zip(user_counts_queued.index,  user_counts_queued)]
user_counts_list_running_singlenode   = [(user, count) for user, count in zip(user_counts_running_singlenode.index,   user_counts_running_singlenode)]
user_counts_list_running_multiplenode = [(user, count) for user, count in zip(user_counts_running_multiplenode.index, user_counts_running_multiplenode)]

# per user
total_running_nodes_by_user = running.groupby('USER')['NODES'].sum()
total_queued_nodes_by_user  = queued.groupby('USER')['NODES'].sum()
total_running_nodes = total_running_nodes_by_user.sum()
total_queued_nodes  = total_queued_nodes_by_user.sum()
sn_r_by_user = singlen_running.groupby('USER')['NODES'].sum()
mn_r_by_user = multiplen_running.groupby('USER')['NODES'].sum()
sn_q_by_user = singlen_queued.groupby('USER')['NODES'].sum()
mn_q_by_user = multiplen_queued.groupby('USER')['NODES'].sum()
total_sn_r = sn_r_by_user.sum()
total_mn_r = mn_r_by_user.sum()
total_sn_q = sn_q_by_user.sum()
total_mn_q = mn_q_by_user.sum()

# entire system
curr_cap_nodes = total_sn_r + total_mn_r


# print info
print(f'=================================================================')
print(f'                             JOBSHAME                            ')
print(f'=================================================================\n')

print(f'-----------------------------------------------------------------')
print(f'|                            SUMMARY                            |')
print(f'-----------------------------------------------------------------')
print(f'|         |       SINGLE-NODE        |        MULTI-NODE        |')
print(f'|         |  JOBS | NODES | RESOUR.  |   JOBS | NODES | RESOUR. |')
print(f'-----------------------------------------------------------------')
print(f'| RUNNING | {len(singlen_running):<6}| {total_sn_r:<6}| {100*total_sn_r/curr_cap_nodes:<4.1f} %   |   {len(multiplen_running):<5}| {total_mn_r:<6}| {100*total_mn_r/curr_cap_nodes:.1f} %  |')
print(f'| QUEUED  | {len(singlen_queued):<6}| {total_sn_q:<6}| {100*total_sn_q/curr_cap_nodes:<4.1f} %  |   {len(multiplen_queued):<5}| {total_mn_q:<6}| {100*total_mn_q/curr_cap_nodes:<4.1f} % |')
print(f'-----------------------------------------------------------------\n')


print('Top users by total running jobs')
print('-----------------------------------------------------------------')
print('|     USER      | TOTAL RUNNING | PERCENT OF TOTAL | PERCENT OF |')
print('|               |     JOBS      |   RUNNING JOBS   | RESOURCES  |')
print('-----------------------------------------------------------------')
for user, count in user_counts_list_running[:10]:
    perctotal = 100*count/len(running)
    percres   = 100*total_running_nodes_by_user[user]/total_running_nodes
    print(f'| {user:<14}| {count:<14}| {perctotal:<17.1f}| {percres:<11.1f}|')
print('-----------------------------------------------------------------\n')

print(' Top users running single-node jobs')
print(' ----------------------------------------------------------------')
print(' |     USER      | SINGLE-NODE  | PERCENT OF TOTAL | PERCENT OF |')
print(' |               | RUNNING JOBS |   RUNNING JOBS   | RESOURCES  |')
print(' ----------------------------------------------------------------')
for user, count in user_counts_list_running_singlenode[:8]:
    perctotal = 100*count/len(running)
    percres   = 100*sn_r_by_user[user]/curr_cap_nodes
    print(f' | {user:<14}| {count:<13}| {perctotal:<17.1f}| {percres:<11.1f}|')
print(' ----------------------------------------------------------------\n')


print(' Top users running multi-node jobs')
print(' ----------------------------------------------------------------')
print(' |               |  MULTI-NODE  |    PERCENT OF    |  PERCENT   |')
print(' |     USER      |   RUNNING    |   TOTAL RUNNING  |     OF     |')
print(' |               | JOBS | NODES |       JOBS       | RESOURCES  |')
print(' ----------------------------------------------------------------')
for user, count in user_counts_list_running_multiplenode[:8]:
    perctotal = 100*count/len(running)
    percres   = 100*mn_r_by_user[user]/curr_cap_nodes
    nnodes    = mn_r_by_user[user]
    print(f' | {user:<14}| {count:<5}| {nnodes:<6}| {perctotal:<17.1f}| {percres:<11.1f}|')
print(' ----------------------------------------------------------------\n')


print('Top users with queued jobs')
print('-----------------------------------------------------------------')
print('|     USER      | TOTAL QUEUED | TOTAL NUMBER | PERCENT OF CURR |')
print('|               |    JOBS      |   OF NODES   | SYSTEM CAPACITY |')
print('-----------------------------------------------------------------')
for user, count in user_counts_list_queued[:8]:
    totalnodes = total_queued_nodes_by_user[user]
    percres    = 100*total_queued_nodes_by_user[user]/curr_cap_nodes
    print(f'| {user:<14}| {count:<13}| {totalnodes:<13.1f}| {percres:<16.1f}|')
print('-----------------------------------------------------------------\n')

