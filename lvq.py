from random import random, shuffle


def find_winner(w, vec):
    m = (w[0][0]-vec[0])**2 + (w[0][1]-vec[1])**2 + (w[0][2]-vec[2])**2 + (w[0][3]-vec[3])**2
    ii = 0
    for i in range(len(w)):
        d = (w[i][0]-vec[0])**2 + (w[i][1]-vec[1])**2 + (w[i][2]-vec[2])**2 + (w[i][3]-vec[3])**2
        if d < m:
            m = d
            ii = i
    return ii


def update_weights(w, vec, winner_index, alpha, label):
    if winner_index != label:
        alpha *= -1
    for i in range(len(w[winner_index])):
        w[winner_index][i] += alpha * (vec[i] - w[winner_index][i])
    return winner_index == label


def L(alpha, count):
    return alpha * 0.75**count


def train(w, train_vectors, labels):
    epoch = 0
    while epoch < 200:
        check = True
        for i in range(len(train_vectors)):
            vec = train_vectors[i]
            label = labels[i]
            winner = find_winner(w, vec)
            c = update_weights(w, vec, winner, L(ALPHA, epoch), label)
            if not c:
                check = False
        if check:
            break
        epoch += 1


def predict(w, test_vector):
    return find_winner(w, test_vector)


def score(w, test_vectors, labels):
    correct = 0
    count = 0
    for i in range(len(test_vectors)):
        vec = test_vectors[i]
        label = labels[i]
        prediction = predict(w, vec)
        count += 1
        if prediction == label:
            correct += 1
    return correct / count


if __name__ == "__main__":
    ALPHA = 0.9
    w = []
    for i in range(3):
        a = random()/5
        b = random()/5
        c = random()/5
        e = random()/5
        w.append([a, b, c, e])
    s = []
    y_out = []
    with open("iris.data", 'r') as file:
        total = file.read().split()
        for sample in total:
            data = sample.split(',')
            s.append([float(data[0]), float(data[1]), float(data[2]), float(data[3])])
            if data[4] == 'Iris-setosa':
                y_out.append(0)
            if data[4] == 'Iris-versicolor':
                y_out.append(1)
            if data[4] == 'Iris-virginica':
                y_out.append(2)
    train_xx = s[:40] + s[50: 90] + s[100: 140]
    train_yy = y_out[:40] + y_out[50: 90] + y_out[100: 140]
    test_x = s[40:50] + s[90:100] + s[140:]
    test_y = y_out[40:50] + y_out[90:] + y_out[140:]
    li = list(range(len(train_xx)))
    shuffle(li)
    train_x = train_xx.copy()
    train_y = train_yy.copy()
    for i in range(len(train_xx)):
        train_x[li[i]] = train_xx[i]
        train_y[li[i]] = train_yy[i]

    train(w, train_x, train_y)
    print("W:", w)
    acc = score(w, test_x, test_y)
    print("accuracy:", acc*100, '%')
