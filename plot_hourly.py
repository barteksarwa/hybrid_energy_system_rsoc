import matplotlib.pyplot as plt
import calendar
import sys
sys.path.append("..")

import InitParams

max = [2.115189384, 2.115189384, 2.013014697, 2.013014697, 1.398115946, 1.398115946, 1.174805481, 1.174805481, 1.398115946, 1.398115946, 2.115189384, 2.115189384]
min = [0.193053601, 0.193053601, 0.156862951, 0.128151315, 0.128151315, 0.128151315, 0.128151315, 0.128151315, 0.128151315, 0.128151315, 0.193053601, 0.193053601]
avg = [0.534124886, 0.54009008, 0.512093015, 0.482908281, 0.454159443, 0.431561187, 0.405332577, 0.405332577, 0.456788532, 0.491777661, 0.526225654, 0.535516176]

max = [x * 1000 for x in max]
min = [x * 1000 for x in min]
avg = [x * 1000 for x in avg]

labels = list(calendar.month_name)[1:]

plt.plot(labels, max, label='Max', c='#845EC2')
plt.plot(labels, min, label='Min', c='#FF6F91')
plt.plot(labels, avg, label='Avg', c='#FFC75F')

# plt.xlabel('Month')
plt.ylabel('Hourly load demand [W]')
# plt.title('Hourly electric load demand range for each month')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
    fancybox=True, shadow=True, ncol=3)
plt.xticks(rotation=45, ha='right')
plt.minorticks_off()
# plt.show()
plt.savefig('fig3.pdf', format='pdf', bbox_inches='tight')