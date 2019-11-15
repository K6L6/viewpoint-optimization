import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import os

from scipy import signal

# directory_path = "./train_result/4_sm2018_d0.01_b30_e400_gs12_testv0/"
# directory_path = "./train_result/4_sm2018_d0.01_b30_e400_gs12_Vb4Gauss/"
# file_name = "variance_10.hdf5"

def max_var_ofiles(directory):
    all_var = []
    for file_n in os.listdir(directory):
        if file_n.endswith(".hdf5"):
            with h5py.File(directory+file_n) as f:
                var1 = np.array(f["variance_b4_gaussian"])
                var2 = np.array(f["variance_all_viewpoints"])
                
                all_var.extend(np.concatenate((var1,var2),axis=None))
            
    return np.max(all_var), np.min(all_var)   

def main():
    file_number = 1

    # y_max, y_min = max_var_ofiles(args.directory_path)
    all_var_bg = []
    all_var_ag = []
    var_labels = []

    for filename in os.listdir(args.directory_path):
        if filename.endswith(".hdf5"):
            with h5py.File(args.directory_path+filename) as f:
                print(f)
                obs_viewpoint = np.array(f["observed_viewpoint"])
                obs_angle = np.array(f["obs_viewpoint_horizontal_angle"])
                x = np.array(np.squeeze(f["query_viewpoints"]))
                y_ag = np.array(f["variance_all_viewpoints"])
                y_bg = np.array(f["variance_b4_gaussian"])
                var_z = np.array(f["variance_of_z"])
                c = np.array(f["c"])

                # ipdb.set_trace()
                # x = x[:129]
                x = np.arange(0,360,360/129)
                y_bg = y_bg[:129]
                y_ag = y_ag[:129]
                var_z = var_z[:129]
                var_c = c[:129]

                y_mean_BG = []
                for i in range(len(y_bg)):
                    y_mean_BG.append(np.mean(y_bg[i],axis=None))

                y_mean_AG = []
                for i in range(len(y_ag)):
                    y_mean_AG.append(np.mean(y_ag[i],axis=None))
                
                z_mean = []
                for i in range(len(var_z)):
                    z_mean.append(np.mean(var_z[i],axis=None))
                
                c_mean = []
                for i in range(len(var_c)):
                    c_mean.append(np.mean(var_c[i],axis=None))

                y1=y_mean_BG
                y2=y_mean_AG
                # y2=y_sum_AG
                y3=z_mean
                y4=c_mean
                # ipdb.set_trace()
                # y1 = y_bg + (np.mean(y_ag)-np.mean(y_bg))/2
                # y2 = y_ag - (np.mean(y_ag)-np.mean(y_bg))/2
                # y3 = var_z/1000
                fig, ax1 = plt.subplots()

                color = 'tab:blue'
                ax1.set_xlabel('angle (degrees)')
                ax1.set_ylabel('variance before Gaussian', color=color)
                ln1 = ax1.plot(x,y1, 'b', label='mean of latent variance',linewidth=0.5)
                
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_xticks(np.arange(0,361, step=30))

                ax2 = ax1.twinx()
                color = 'tab:red'
                ax2.set_ylabel('variance after Gaussian', color=color)  # we already handled the x-label with ax1
                # ln2 = ax2.plot(x,y2, 'r', label='mean pixel variance of 100 predictions',linewidth=1.0)
                # ln3 = ax2.plot(x,y3, 'r', label='average variance of Z', linewidth=0.5)
                # ln4 = ax2.plot(x,signal.savgol_filter(y3,53,3),'r',linestyle='dashed',label='Trend curve')
                ln5 = ax2.plot(x,y4,'r',label='average variance of c',linewidth=0.5)
                ax2.tick_params(axis='y', labelcolor=color)
                
                # lns = ln1 + ln2 + ln3
                lns = ln1 + ln5
                labs = [l.get_label() for l in lns]
                ax1.legend(lns, labs)
                # plt.grid(b=True,which='both',axis='both')
                plt.tight_layout()
                plt.title("viewpoint "+str(obs_angle)+" degrees")
                fig.savefig(args.save_path+"VarianceAgainstC_"+filename.split(".")[0]+".png",bbox_inches="tight")
                plt.close(fig)

                # all_var_bg.append(y_bg)
                # all_var_ag.append(y_mean_AG)
                # var_labels.append(filename.split(".")[0])
                file_number += 1
                # input()
                # if file_number==2:
                #     break
        plt.close()
    
    # Plot all in one
    # y = np.stack(all_var_bg,axis=1)
    # lines = plt.plot(x,y)
    # plt.legend(lines,var_labels)
    # plt.xlabel("angle (degrees)")
    # plt.ylabel("variance")
    # plt.savefig(args.save_path+"b4Gauss_all")
    # plt.clf()
    # plt.close()

    # y = np.stack(all_var_ag,axis=1)
    # lines = plt.plot(x,y)
    # plt.legend(lines,var_labels)
    # plt.xlabel("angle (degrees)")
    # plt.ylabel("variance")
    # plt.savefig(args.save_path+"afterGauss_all")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory-path","-dir", type=str, required=True)
    parser.add_argument("--save-path","-saveto",type=str,required=True)
    args = parser.parse_args()
    main()
    

