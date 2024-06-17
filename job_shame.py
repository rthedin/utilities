import pandas as pd
import subprocess

# get queue and format it
command = "squeue -o '%.15u %.8a %.20P %3j %.2t %.10M %.5D %.4C  %4r' -S V"
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


# Futher split between regular and standby
singlen_running_stdby   =   singlen_running[  singlen_running['PARTITION'].str.endswith('stdby')]
multiplen_running_stdby = multiplen_running[multiplen_running['PARTITION'].str.endswith('stdby')]
singlen_queued_stdby    =   singlen_queued[  singlen_queued['PARTITION'].str.endswith('stdby')]
multiplen_queued_stdby  = multiplen_queued[multiplen_queued['PARTITION'].str.endswith('stdby')]
# Per user per standby
sn_r_by_user_stdby = singlen_running_stdby.groupby('USER')['NODES'].sum()
mn_r_by_user_stdby = multiplen_running_stdby.groupby('USER')['NODES'].sum()
sn_q_by_user_stdby = singlen_queued_stdby.groupby('USER')['NODES'].sum()
mn_q_by_user_stdby = multiplen_queued_stdby.groupby('USER')['NODES'].sum()


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
print(f'┌───────────────────────────────────────────────────────────────────────┐')
print(f'│                                JOBSHAME                               │')
print(f'└───────────────────────────────────────────────────────────────────────┘\n')

print(f'┌───────────────────────────────────────────────────────────────────────┐')
print(f'│                                 SUMMARY                               │')
print(f'├────────────────┬───────────────────────────┬──────────────────────────┤')
print(f'│                │       SINGLE-NODE         │        MULTI-NODE        │')
print(f'│                │                           │                          │')
print(f'│                │  JOBS  │ NODES │ RESOUR.  │  JOBS  │ NODES │ RESOUR. │')
print(f'├────────────────┼────────┼───────┼──────────┼────────┼───────┼─────────┤')
print(f'│    RUNNING     │ {len(singlen_running):<7}│ {total_sn_r:<6}│ {100*total_sn_r/curr_cap_nodes:<4.1f}  %  │   {len(multiplen_running):<5}│ {total_mn_r:<6}│ {100*total_mn_r/curr_cap_nodes:.1f}  % │')
print(f'│    QUEUED      │ {len(singlen_queued):<7}│ {total_sn_q:<6}│ {100*total_sn_q/curr_cap_nodes:<5.1f} %  │   {len(multiplen_queued):<5}│ {total_mn_q:<6}│ {100*total_mn_q/curr_cap_nodes:<5.1f} % │')
print(f'└────────────────┴────────┴───────┴──────────┴────────┴───────┴─────────┘\n\n')


print(' Top users by total running jobs')
print(f'┌─────────────────┬──────────────────┬───────────────────┬──────────────┐')
print(f'│      USER       │   TOTAL RUNNING  │  PERCENT OF TOTAL │   RESOURCES  │')
print(f'│                 │   JOBS (STANDBY) │    RUNNING JOBS   │   (PERCENT)  │')
print(f'├─────────────────┼──────────────────┼───────────────────┼──────────────┤')
for user, count in user_counts_list_running[:10]:
    perctotal = 100*count/len(running)
    percres   = 100*total_running_nodes_by_user[user]/total_running_nodes
    if user in sn_r_by_user_stdby or user in mn_r_by_user_stdby:
        stdby_tot = 0
        if user in sn_r_by_user_stdby: stdby_tot += sn_r_by_user_stdby[user]
        if user in mn_r_by_user_stdby: stdby_tot += mn_r_by_user_stdby[user]
        stdby_tot = f'({stdby_tot})'
        print(f'│ {user:<16}│ {count:<7} {stdby_tot:<9}│ {perctotal:<18.1f}│ {percres:<13.1f}│')
    else:
        print(f'│ {user:<16}│ {count:<17}│ {perctotal:<18.1f}│ {percres:<13.1f}│')
print(f'└─────────────────┴──────────────────┴───────────────────┴──────────────┘\n')


print(f'    Top users running single-node jobs')
print(f'   ┌───────────────┬──────────────────┬──────────────────┬───────────┐')
print(f'   │     USER      │   SINGLE-NODE    │ PERCENT OF TOTAL │ RESOURCES │')
print(f'   │               │   RUNNING JOBS   │   RUNNING JOBS   │ (PERCENT) │')
print(f'   ├───────────────┼──────────────────┼──────────────────┼───────────┤')
for user, count in user_counts_list_running_singlenode[:8]:
    perctotal = 100*count/len(running)
    percres   = 100*sn_r_by_user[user]/curr_cap_nodes
    if user in sn_r_by_user_stdby:
        stdby_tot = sn_r_by_user_stdby[user]
        stdby_tot = f'({stdby_tot})'
        print(f'   │ {user:<14}│ {count:<7} {stdby_tot:<9}│ {perctotal:<17.1f}│ {percres:<10.1f}│')
    else:
        print(f'   │ {user:<14}│ {count:<17}│ {perctotal:<17.1f}│ {percres:<10.1f}│')
print(f'   └───────────────┴──────────────────┴──────────────────┴───────────┘\n')


print(f'    Top users running multi-node jobs')
print(f'   ┌───────────────┬──────────────────┬──────────────────┬───────────┐')
print(f'   │               │    MULTI-NODE    │    PERCENT OF    │ RESOURCES │')
print(f'   │     USER      │     RUNNING      │   TOTAL RUNNING  │ (PERCENT) │')
print(f'   │               │ JOBS │   NODES   │       JOBS       │           │')
print(f'   ├───────────────┼──────┼───────────┼──────────────────┼───────────┤')
for user, count in user_counts_list_running_multiplenode[:8]:
    perctotal = 100*count/len(running)
    percres   = 100*mn_r_by_user[user]/curr_cap_nodes
    nnodes    = mn_r_by_user[user]
    if user in mn_r_by_user_stdby:
        stdby_tot = mn_r_by_user_stdby[user]
        stdby_tot = f'({stdby_tot})'
        print(f'   │ {user:<14}│ {count:<5}│ {nnodes:<4} {stdby_tot:<5}│ {perctotal:<17.1f}│ {percres:<10.1f}│')
    else:
        print(f'   │ {user:<14}│ {count:<5}│ {nnodes:<10}│ {perctotal:<17.1f}│ {percres:<10.1f}│')
print(f'   └───────────────┴──────────────────┴──────────────────┴───────────┘\n')


print(f' Top users with queued jobs')
print(f'┌─────────────────┬────────────────┬─────────────────┬──────────────────┐')
print(f'│      USER       │  TOTAL QUEUED  │ TOTAL NUMBER OF │  PERCENT OF CURR │')
print(f'│                 │       JOBS     │ NODES (STANDBY) │  SYSTEM CAPACITY │')
print(f'├─────────────────┼────────────────┼─────────────────┼──────────────────┤')
for user, count in user_counts_list_queued[:8]:
    totalnodes = total_queued_nodes_by_user[user]
    percres    = 100*total_queued_nodes_by_user[user]/curr_cap_nodes
    if user in sn_q_by_user_stdby or user in mn_q_by_user_stdby:
        stdby_tot = 0
        if user in sn_q_by_user_stdby: stdby_tot += sn_q_by_user_stdby[user]
        if user in mn_q_by_user_stdby: stdby_tot += mn_q_by_user_stdby[user]
        stdby_tot = f'({stdby_tot})'
        print(f'│ {user:<16}│ {count:<15}│ {int(totalnodes):<7} {stdby_tot:<8}│ {percres:<17.1f}│')
    else:
        print(f'│ {user:<16}│ {count:<15}│ {int(totalnodes):<16}│ {percres:<17.1f}│')
print(f'└─────────────────┴────────────────┴─────────────────┴──────────────────┘\n')
