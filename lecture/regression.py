# 해당 소스는 Andrew NG 의 머신러닝 강좌의 cost function 설명부분을 참고하여 작성함
# 소스는 최적화 보다 강의 내용을 이해하기 위한 목적으로 비효율적인 부분도 내용에 맞추어서 작성
# python3 를 기준으로 작성한다



## 가설함수 Θ0와 Θ1 을 고정시킨다
def hypothesis(x, theta0, theta1):
    return x * theta1 + theta0

### 비용함수
def jcost(examples, theta0, theta1) :
    cost = 0.0
    m = len(examples)
    for x, y in examples :
        result = hypothesis(x, theta0, theta1)
        cost = cost + (((y - result) ** 2.0))

    return cost / m


### 실제로 강의 내용처럼  Θ0와  Θ1 을 맞추어서 작성한다.
### cost function 을 미분하는 결과 함수를 개별적으로 작성하는것은 비효율적이지만 여기서는 고정시켜서 작성한다
def jcost_derivative_by_x(examples, theta0, theta1) :
    cost = 0.0
    m = len(examples)
    for x, y in examples :
        result = hypothesis(x, theta0, theta1)
        cost = cost + (result - y) * x

    return cost /m


def jcost_derivative_by(examples, theta0, theta1) :
    cost = 0.0
    m = len(examples)
    for x, y in examples :
        result = hypothesis(x, theta0, theta1)
        cost = cost + (result - y)

    return cost / m



### gradient descent 구현
### 여기서는 iteration 인자를 사용하여 반복 할수 있는 구조로 구현된다
### 이처럼 iteration 인자를 이용하여 cost 값을 확인 하며 학습 시킬수 잇으며 아니면 cost 값이 특정 기준치 이하 일때만 학습 시킬수 있다

def gradient_descent(examples, init_theta0, init_theta1, alpha, iteri = 1000) :
    m = len(examples)

    theta0 = init_theta0
    theta1 = init_theta1

    for index in range(iteri) :
        cost = jcost_derivative_by(examples, theta0, theta1)
        theta0  = theta0 - alpha * cost

        cost = jcost_derivative_by_x(examples, theta0, theta1)
        theta1 = theta1 - alpha * cost

    return theta0, theta1, jcost(examples, theta0, theta1)


examples = [(1,3), (2,5), (3, 7), (4, 9)]
results = gradient_descent(examples, 300, 599, 0.1)
print(results)


