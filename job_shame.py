import pandas as pd
import subprocess

# get queue and format it
command = "squeue -o '%.15u %.8a %.15P %3j %.2t %.10M %.5D %.4C  %4r' -S V"
result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
data = [line.split() for line in result.stdout.split('\n') if line.strip()]
df = pd.DataFrame(data[1:], columns=data[0])

# split running and queued jobs
running = df[df['ST'] == 'R']
queued  = df[df['ST'] == 'PD']
singlen_running = running[running['NODES'] == '1']
singlen_queued = queued[queued['NODES'] == '1']
multiplen_running = running[running['NODES'] != '1']
multiplen_queued = queued[queued['NODES'] != '1']
user_counts_running = running['USER'].value_counts()
user_counts_queued = queued['USER'].value_counts()
user_counts_running_singlenode   = singlen_running['USER'].value_counts()
user_counts_running_multiplenode = multiplen_running['USER'].value_counts()
user_counts_list_running = [(user, count) for user, count in zip(user_counts_running.index, user_counts_running)]
user_counts_list_queued = [(user, count) for user, count in zip(user_counts_queued.index, user_counts_queued)]
user_counts_list_running_singlenode = [(user, count) for user, count in zip(user_counts_running_singlenode.index, user_counts_running_singlenode)]
user_counts_list_running_multiplenode = [(user, count) for user, count in zip(user_counts_running_multiplenode.index, user_counts_running_multiplenode)]

# print info
print(f'{len(df)} total jobs; {len(running)} are running, {len(queued)} are queued')
print(f'Out of {len(running)} running jobs, {len(singlen_running)} are single node, {len(multiplen_running)} are multi-node')
print(f'Out of {len(queued)} queued jobs, {len(singlen_queued)} are single node, {len(multiplen_queued)} are multi-node')

print(f'\nTop users with total running jobs')
for user, count in user_counts_list_running[:8]:
    print(f"User: {user},      \t total number of running jobs: {count} ({100*count/len(running):.1f}%)")

print(f'\n   Top users with running single-node jobs')
for user, count in user_counts_list_running_singlenode[:8]:
    print(f"   User: {user},    \t number of single-node running jobs: {count} ({100*count/len(running):.1f}%)")
print(f'\n   Top users with running multiple-node jobs')
for user, count in user_counts_list_running_multiplenode[:8]:
    print(f"   User: {user},    \t number of multiple-node running jobs: {count} ({100*count/len(running):.1f}%)")

print(f'\nTop users with queued jobs')
for user, count in user_counts_list_queued[:8]:
    print(f"User: {user},\t total number of queued jobs: {count}")
