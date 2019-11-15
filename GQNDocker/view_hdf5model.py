import argparse
import h5py

def main():
    #==============================================================================
    # Prints layer number and named links in model.hdf5
    #==============================================================================
    with h5py.File(args.model_directory) as f: 
        index = list(f) 
        for i in index: 
            layers = f[i] 
            namedlinks = [x for x in layers] 
            dict = {"layer":i,"namedlinks":namedlinks} 
            print(dict) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-directory", "-model", type=str, required=True)
    args = parser.parse_args()
    main()