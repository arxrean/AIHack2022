import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import shutil
import sys
import scipy.stats
import scipy.signal
import scipy.optimize

def get_one_image(filename, timestep, index, ranges=None):
    """
    Get a single 3D image from the HDF5 container.
    (Accessing them in batch is much more efficient.

    filename to read from
    timestep (0-9) of the time evolution
    index instance of the time evolution
    """
    with h5py.File(filename, "r") as file_handle:
        data = file_handle["time"+str(timestep)][index]
    if ranges != None:
        return data[ranges, ranges, ranges]
    else:
        return data


def get_structure_factor(data, N=100):
    hat_data = np.fft.fftn(data)
    hat_data = np.fft.fftshift(hat_data)
    hat_data = np.abs(hat_data**2)

    center = np.asarray(data.shape)/2
    idx = np.indices(data.shape)
    q = np.sqrt(np.sum((idx - center[:, np.newaxis, np.newaxis, np.newaxis])**2, axis=0))
    max_q = np.max(q)

    q_array = np.linspace(0, max_q, N+1)
    sq_array = np.zeros(N)

    for i in range(len(sq_array)):
        sq_array[i] = np.mean(hat_data[np.logical_and(q_array[i] <= q, q <= q_array[i+1])])

    q_array = q_array[:-1] # + (q_array[1]-q_array[0])/2
    return q_array, sq_array


def plot_sq(q, sq, ax, name="sq.pdf"):
##    fig, ax = plt.subplots()
    ax.set_xlabel("$q$ [arb. units]")
    ax.set_ylabel("$S(q)$ [arb. units]")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
##    ax.plot(q, sq)
    q = q[0]
    sqs = np.array(sq)
    ax.errorbar(q, sqs.mean(axis=0), yerr=sqs.std(axis=0))

##    fig.savefig(name, transparent=True, bbox_inches='tight', pad_inches=0)
##    plt.close(fig)


def exp_decay(t, A, tau):
    return A*np.exp(-t/tau)


def get_correlation_length(q, sq, ax, plot_name=None):
    q = q[0]
    sqs = np.array(sq)
    xis = []
    rs = []
    grs = []
    
    for idx in range(len(sqs)):
        sq = sqs[idx]
        gr = np.fft.irfft(sq)
        gr = (gr[:gr.shape[0]//2] + np.flip(gr[gr.shape[0]//2:]))/2

        gr -= np.mean(gr)
        r = np.linspace(0, np.max(1/q[1:]), gr.shape[0])
        rs.append(r)
        grs.append(gr)
        
        max_idx = scipy.signal.argrelextrema(gr, np.less)[0]
        
        popt, pcov = scipy.optimize.curve_fit(exp_decay, r[max_idx], gr[max_idx], p0=(-3e5, 10))
        xis.append(popt[1])
        
    ##    if plot_name:
    ##        fig, ax = plt.subplots()
        ax.set_xlabel("$r$ [arb. units]")
        ax.set_ylabel("$g(r)$ [arb. units]")

##        ax.plot(r, gr, "o", label="data")
        ax.plot(r[max_idx], gr[max_idx], "o", label="max.")
        ax.plot(r, exp_decay(r, *popt), label="fit")
    rs = np.array(rs)
    grs = np.array(grs)
    ax.errorbar(rs.mean(axis=0), grs.mean(axis=0), yerr=grs.std(axis=0))

##        ax.legend(loc="best")

##        fig.savefig(plot_name, transparent=True, bbox_inches='tight', pad_inches=0)
##        plt.close(fig)
    return popt[1]


def get_volume_fraction(data):
    return data.mean()


def plot_time_evolution(filename, index, keys, ranges, name="time.pdf"):
    times = []
    volumes = []
    qs = []
    sqs = []
    fig, ax = plt.subplots(2,2, figsize=(12,12))
    for time in range(0, len(keys)):
        img = get_one_image(filename, time, index, ranges)
        q, sq = get_structure_factor(img, )
##        print(q, sq)
##        xi = get_correlation_length(q, sq, ax.flatten()[2])
        volume = get_volume_fraction(img)
##        plot_sq(q, sq, ax.flatten()[3])

        times.append(time)
        volumes.append(volume)
##        xis.append(xi)
        qs.append(q)
        sqs.append(sq)
    xis = get_correlation_length(qs, sqs, ax.flatten()[2])
    plot_sq(qs, sqs, ax.flatten()[3])

    
##    fig, ax = plt.subplots()
    ax0 = ax.flatten()[0]
    ax0.set_xlabel("time")
    ax0.set_ylabel("measure")
    ax0.plot(times, volumes, label="volume frac. (const.)")
    ax1 = ax.flatten()[1]
    ax1.set_xlabel("time")
    ax1.set_ylabel("measure")
    ax1.plot(times, xis, label="correlation. (power law incr.)")

    ax1.legend(loc="best")
    plt.suptitle(f"Sample {index}")
    fig.savefig(name, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return times, volumes, xis


def difference(data: "h5py read file", absl=False):
    assert isinstance(data, (h5py.File, np.ndarray)), "wrong data type..."
    keys = sorted(list(f.keys()), key=lambda key: int(key.split("time")[-1]))
    sample_counts = data[keys[0]].shape[0] #scalar, how many samples?
    def diff_per_sample(sample_index):
        current = np.stack([data[k][sample_index] for k in keys[1:]], axis=0)
        prev = np.stack([data[k][sample_index] for k in keys[:-1]], axis=0) #(T-1),DDD
        assert prev.shape == current.shape, "both data must have same shape..."
        diff = np.abs(current - prev) if absl else current - prev
        return diff
    diff = list(map(lambda inp: diff_per_sample(inp), np.arange(sample_counts)))
    return diff
    

def main(argv):
    if len(argv) != 3:
        print("./Usage tools.py filename time index")
        return
    filename = argv[0]
    time = int(argv[1])
    index = int(argv[2])

    img = get_one_image(filename, time, index)
    q, sq = get_structure_factor(img)
    xi = get_correlation_length(q, sq, "gr.pdf")
    plot_sq(q, sq)

    plot_time_evolution(filename, index)
                        
######
if __name__ == "__main__":
    roots = "/Users/hyunpark/Downloads/"
    os.chdir(f"{roots}results")
    filename = "alldata.h5"
    f = h5py.File(filename, "r")

    keys = sorted(list(f.keys()), key=lambda key: int(key.split("time")[-1]))
##    ranges = slice(4,-2)
    ranges = None
    times, volumes, xis = plot_time_evolution(filename, 1, keys, ranges)
    if os.path.isfile(f"{roots}time.pdf"): os.remove(f"{roots}time.pdf")
    shutil.move(f"{roots}results/time.pdf", f"{roots}")
##    os.system("mv results/*.pdf /Users/hyunpark/Downloads")
    os.chdir(f"{roots}")

  
