import pandas as pd
import subprocess

# get queue and format it
command = "squeue -o '%.15u %.8a %.20P %3j %.2t %.10M %.5D %.4C  %4r' -S V"
result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
data = [line.split() for line in result.stdout.split('\n') if line.strip()]
df = pd.DataFrame(data[1:], columns=data[0])
df['NODES'] = pd.to_numeric(df['NODES'], errors='coerce')

# split running and queued jobs
r = df[df['ST'] == 'R']
q = df[df['ST'] == 'PD']
sn_r = r[r['NODES'] == 1]
sn_q = q[q['NODES'] == 1]
mn_r = r[r['NODES'] != 1]
mn_q = q[q['NODES'] != 1]
user_count_r = r['USER'].value_counts()
user_count_q = q['USER'].value_counts()
user_count_r_sn = sn_r['USER'].value_counts()
user_count_r_mn = mn_r['USER'].value_counts()
user_count_list_r = [(user, count) for user, count in zip(user_count_r.index, user_count_r)]
user_count_list_q = [(user, count) for user, count in zip(user_count_q.index, user_count_q)]
user_count_list_r_sn = [(user, count) for user, count in zip(user_count_r_sn.index, user_count_r_sn)]
user_count_list_r_mn = [(user, count) for user, count in zip(user_count_r_mn.index, user_count_r_mn)]

# Count per number of nodes for multi-node runs
node_user_count_r_mn = mn_r.groupby('USER').agg({'NODES': 'sum', 'USER': 'count'}).rename(columns={'USER': 'LINES'})
node_user_count_r_mn = node_user_count_r_mn.sort_values(by='NODES', ascending=False)
node_user_count_list_r_mn = list(node_user_count_r_mn.itertuples(index=True, name=None))


# Futher split between regular and standby
sn_r_stdby = sn_r[sn_r['PARTITION'].str.endswith('stdby')]
mn_r_stdby = mn_r[mn_r['PARTITION'].str.endswith('stdby')]
sn_q_stdby = sn_q[sn_q['PARTITION'].str.endswith('stdby')]
mn_q_stdby = mn_q[mn_q['PARTITION'].str.endswith('stdby')]
# Per user per standby
sn_r_by_user_stdby = sn_r_stdby.groupby('USER')['NODES'].sum()
mn_r_by_user_stdby = mn_r_stdby.groupby('USER')['NODES'].sum()
sn_q_by_user_stdby = sn_q_stdby.groupby('USER')['NODES'].sum()
mn_q_by_user_stdby = mn_q_stdby.groupby('USER')['NODES'].sum()

# per user
tot_r_nodes_by_user = r.groupby('USER')['NODES'].sum()
tot_q_nodes_by_user = q.groupby('USER')['NODES'].sum()
tot_r_nodes = tot_r_nodes_by_user.sum()
tot_q_nodes = tot_q_nodes_by_user.sum()
sn_r_by_user = sn_r.groupby('USER')['NODES'].sum()
mn_r_by_user = mn_r.groupby('USER')['NODES'].sum()
sn_q_by_user = sn_q.groupby('USER')['NODES'].sum()
mn_q_by_user = mn_q.groupby('USER')['NODES'].sum()
tot_sn_r = sn_r_by_user.sum()
tot_mn_r = mn_r_by_user.sum()
tot_sn_q = sn_q_by_user.sum()
tot_mn_q = mn_q_by_user.sum()

# entire system
curr_cap_nodes = tot_sn_r + tot_mn_r
perc_res_sn_r = 100*tot_sn_r/curr_cap_nodes
perc_res_sn_q = 100*tot_sn_q/curr_cap_nodes
perc_res_mn_r = 100*tot_mn_r/curr_cap_nodes
perc_res_mn_q = 100*tot_mn_q/curr_cap_nodes


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
print(f"│    RUNNING     │ {len(sn_r):<7}│ {tot_sn_r:<6}│ {str(f'{perc_res_sn_r:.1f}') + ' %':<9}│"+
                      f"   {len(mn_r):<5}│ {tot_mn_r:<6}│ {str(f'{perc_res_mn_r:.1f}') + ' %':<8}│")
print(f"│    QUEUED      │ {len(sn_q):<7}│ {tot_sn_q:<6}│ {str(f'{perc_res_sn_q:.1f}') + ' %':<9}│"+
                      f"   {len(mn_q):<5}│ {tot_mn_q:<6}│ {str(f'{perc_res_mn_q:.1f}') + ' %':<8}│")
print(f'└────────────────┴────────┴───────┴──────────┴────────┴───────┴─────────┘\n\n')


print(' Top users by total running jobs')
print(f'┌─────────────────┬──────────────────┬───────────────────┬──────────────┐')
print(f'│      USER       │   TOTAL RUNNING  │  PERCENT OF TOTAL │   RESOURCES  │')
print(f'│                 │   JOBS (STANDBY) │    RUNNING JOBS   │   (PERCENT)  │')
print(f'├─────────────────┼──────────────────┼───────────────────┼──────────────┤')
for user, count in user_count_list_r[:10]:
    perctot = 100*count/len(r)
    percres   = 100*tot_r_nodes_by_user[user]/tot_r_nodes
    if user in sn_r_by_user_stdby or user in mn_r_by_user_stdby:
        stdby_tot = 0
        if user in sn_r_by_user_stdby: stdby_tot += sn_r_by_user_stdby[user]
        if user in mn_r_by_user_stdby: stdby_tot += mn_r_by_user_stdby[user]
        stdby_tot = f'({stdby_tot})'
        print(f'│ {user:<16}│ {count:<7} {stdby_tot:<9}│ {perctot:<18.1f}│ {percres:<13.1f}│')
    else:
        print(f'│ {user:<16}│ {count:<17}│ {perctot:<18.1f}│ {percres:<13.1f}│')
print(f'└─────────────────┴──────────────────┴───────────────────┴──────────────┘\n')


print(f'    Top users running single-node jobs')
print(f'   ┌───────────────┬──────────────────┬──────────────────┬───────────┐')
print(f'   │     USER      │   SINGLE-NODE    │ PERCENT OF TOTAL │ RESOURCES │')
print(f'   │               │   RUNNING JOBS   │   RUNNING JOBS   │ (PERCENT) │')
print(f'   ├───────────────┼──────────────────┼──────────────────┼───────────┤')
for user, count in user_count_list_r_sn[:8]:
    perctot = 100*count/len(r)
    percres   = 100*sn_r_by_user[user]/curr_cap_nodes
    if user in sn_r_by_user_stdby:
        stdby_tot = sn_r_by_user_stdby[user]
        stdby_tot = f'({stdby_tot})'
        print(f'   │ {user:<14}│ {count:<7} {stdby_tot:<9}│ {perctot:<17.1f}│ {percres:<10.1f}│')
    else:
        print(f'   │ {user:<14}│ {count:<17}│ {perctot:<17.1f}│ {percres:<10.1f}│')
print(f'   └───────────────┴──────────────────┴──────────────────┴───────────┘\n')


print(f'    Top users running multi-node jobs')
print(f'   ┌───────────────┬──────────────────┬──────────────────┬───────────┐')
print(f'   │               │    MULTI-NODE    │    PERCENT OF    │ RESOURCES │')
print(f'   │     USER      │     RUNNING      │   TOTAL RUNNING  │ (PERCENT) │')
print(f'   │               │ JOBS │   NODES   │       JOBS       │           │')
print(f'   ├───────────────┼──────┼───────────┼──────────────────┼───────────┤')
for user, ncount, jcount in node_user_count_list_r_mn[:8]:
    perctot = 100*jcount/len(r)
    percres   = 100*mn_r_by_user[user]/curr_cap_nodes
    nnodes    = mn_r_by_user[user]
    if user in mn_r_by_user_stdby:
        stdby_tot = mn_r_by_user_stdby[user]
        stdby_tot = f'({stdby_tot})'
        print(f'   │ {user:<14}│ {jcount:<5}│ {nnodes:<4} {stdby_tot:<5}│ {perctot:<17.1f}│ {percres:<10.1f}│')
    else:
        print(f'   │ {user:<14}│ {jcount:<5}│ {nnodes:<10}│ {perctot:<17.1f}│ {percres:<10.1f}│')
print(f'   └───────────────┴──────────────────┴──────────────────┴───────────┘\n')


print(f' Top users with queued jobs')
print(f'┌─────────────────┬────────────────┬─────────────────┬──────────────────┐')
print(f'│      USER       │  TOTAL QUEUED  │ TOTAL NUMBER OF │  PERCENT OF CURR │')
print(f'│                 │       JOBS     │ NODES (STANDBY) │  SYSTEM CAPACITY │')
print(f'├─────────────────┼────────────────┼─────────────────┼──────────────────┤')
for user, count in user_count_list_q[:8]:
    totnodes = tot_q_nodes_by_user[user]
    percres    = 100*tot_q_nodes_by_user[user]/curr_cap_nodes
    if user in sn_q_by_user_stdby or user in mn_q_by_user_stdby:
        stdby_tot = 0
        if user in sn_q_by_user_stdby: stdby_tot += sn_q_by_user_stdby[user]
        if user in mn_q_by_user_stdby: stdby_tot += mn_q_by_user_stdby[user]
        stdby_tot = f'({stdby_tot})'
        print(f'│ {user:<16}│ {count:<15}│ {int(totnodes):<7} {stdby_tot:<8}│ {percres:<17.1f}│')
    else:
        print(f'│ {user:<16}│ {count:<15}│ {int(totnodes):<16}│ {percres:<17.1f}│')
print(f'└─────────────────┴────────────────┴─────────────────┴──────────────────┘\n')
