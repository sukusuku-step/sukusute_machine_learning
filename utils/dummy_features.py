import numpy as np

def create_dummy_data(N):

    X = np.zeros((N,9))
    y = np.zeros(N)

    for i in range(N):
        # それぞれの値は乱数を使う
        step = np.random.randint(0,15000)
        step_rate = np.random.uniform(-0.5,0.5)
        diff = np.random.randint(-5000,5000)
        roll = np.random.uniform(-30,30)
        pitch = np.random.uniform(-30,30)
        yaw = np.random.uniform(-180,180)
        droll = np.random.uniform(0,5)
        dpitch = np.random.uniform(0,5)
        dyaw = np.random.uniform(0,10)

        # 現在の歩数 : step
        # 歩数変化率 : step_rate
        # 過去平均との差 : diff
        # X軸中心姿勢 : roll
        # Y軸中心姿勢 : pitch
        # Z軸中心姿勢 : yaw
        # X軸中心姿勢変化率 : droll
        # Y軸中心姿勢変化率 : dpitch
        # Z軸中心姿勢変化率 : dyaw
        
        # 9次元の特徴量ベクトルを作成
        X[i] = [
            step,
            step_rate,
            diff,
            roll,
            pitch,
            yaw,
            droll,
            dpitch,
            dyaw
        ]

        # ダミーラベル
        if step < 1000:
            y[i] = 1
        elif droll > 4:
            y[i] = 2
        else:
            y[i] = 0

    return X, y