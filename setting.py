from usegpu import *

# 훈련 함수
def train(model, criterion, optimizer, train_loader):
    device = UseGPU()
    model.train()
    for i, batch in enumerate(train_loader):
        x_train, y_train = batch.text.to(device), batch.label.to(device)
        y_train.data.sub(1) # label 값을 0과 1로 변환

        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = criterion(hypothesis, y_train)
        loss.backward()
        optimizer.step()

# 평가 함수
def evaluate(model, criterion, validation_loader):
    device = UseGPU()
    model.eval()
    corrects, total_loss = 0, 0
    for i, batch in enumerate(validation_loader):
        x_validation, y_validation = batch.text.to(device), batch.label.to(device)
        y_validation.data.sub(1) # label 값을 0과 1로 변환

        prediction = model(x_validation)
        loss = criterion(prediction, y_validation)
        total_loss += loss.item()
        corrects += (prediction.max(1)[1].view(y_validation.size(0)).data == y_validation.data).sum()

    size = len(validation_loader.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size

    return avg_loss, avg_accuracy

