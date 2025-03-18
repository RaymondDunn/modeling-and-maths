#####################################################################
######  test out frequency component neuron activity toy example ####
#####################################################################

%matplotlib qt
sns.set_style('white')
sns.set_context('talk')

# FFT transform testing via https://www.mkdynamics.net/current_projects/2sine_wv_XHz.html
# make a few sine waves of slightly different frequencies... shift them
# xstart, xend = [0, 1]
freq_list = [0.3, 0.5]
sample_rate = 1000 # sample rate in Hertz
num_periods = 40 # number of periods of the sine waves
num_samples = sample_rate * num_periods # total number of samples
amplitude_list = [5, 10]

# define a size
add_shift = True
shift_size_periods = 5
shift_size_samples = shift_size_periods * sample_rate
shift_start_periods = 10
shift_start_samples = shift_start_periods * sample_rate

# function to calculate sine at a particular moment
def calc_sine(x, freq, amplitude):
    return amplitude * np.sin(freq * 2 * np.pi * x)

plot_dict_list = []
xvals = np.linspace(0, num_periods, num_samples)
for i in range(len(freq_list)):
    yvals = calc_sine(xvals, freq_list[i], amplitude_list[i])

    # make minor adjustment and store
    if add_shift:
        segment = yvals[shift_start_samples:len(yvals)-shift_size_samples]
        yvals[shift_start_samples:shift_start_samples + shift_size_samples] = yvals.min()
        yvals[shift_start_samples + shift_size_samples:] = segment

    # add to list fo dicts for easy 
    for j in range(len(xvals)):
        plot_dict_list.append({
            't': xvals[j],
            'y': yvals[j],
            'freq': freq_list[i],
            'amplitude': amplitude_list[i]
        })

plot_df = pd.DataFrame(plot_dict_list)

# make figure
fig = plt.figure()
ax = fig.add_subplot()

# plot
for freq in freq_list:
    df_view = plot_df[plot_df['freq'] == freq]
    xvals = df_view['t']
    yvals = df_view['y']
    ax.plot(xvals, yvals, label='{}Hz'.format(freq))

# also plot combined
xvals = df_view['t']
yvals = np.zeros(len(xvals))
for freq in freq_list:
    yvals += plot_df[plot_df['freq'] == freq]['y'].to_numpy()
    combined = yvals
# plt.plot(xvals, yvals, label='sum of {}Hz'.format(freq_list))

# decorate
ax.legend()
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1,1), fontsize='x-small')
ax.set_xlabel('time')
ax.set_ylabel('magnitude (AU)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

# combine and fft
for freq in freq_list:
    # fft_output = np.fft.rfft(combined)
    fft_output = np.fft.rfft(plot_df[plot_df['freq'] == freq]['y'].to_numpy())
    magnitude = [np.sqrt(i.real**2+i.imag**2)/len(fft_output) for i in fft_output]
    frequencies = [(i*1.0/num_samples)*sample_rate for i in range(num_samples//2+1)]

    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(frequencies, magnitude)
    ax.set_xlim([0, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('amplitude (AU)')
    plt.tight_layout()
