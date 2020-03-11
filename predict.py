import json
from train import predict 

def recover():
    with open('snapshot.txt') as infile:
        snapshot = json.load(infile)
    return snapshot["theta0"], snapshot["theta1"]

def get_user_input():
    mileage = input("What is the mileage of your car ?: ")
    return float(mileage)

def main():
    try:
        mileage = get_user_input()
    except:
        print("This is not a valid value.")
        return -1
    theta0, theta1 = 0, 0
    try:
        theta0, theta1 = recover()
        if type(theta0) is not float or type(theta1) is not float:
            print("Recovered datas are corrupted !")
            theta0, theta1 = 0, 0
    except:
        pass
    prediction = predict(theta0, theta1, mileage)
    print(f"The prediction value is: {prediction}")

if __name__ == "__main__":
    main()