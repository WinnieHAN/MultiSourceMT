import os, sys
def main():
    max_length=50
    a = open(os.path.join('wmt14_3', 'train.de'), 'rb').readlines()
    b = open(os.path.join('wmt14_3', 'train.en'), 'rb').readlines()
    c = open(os.path.join('wmt14_3', 'train.fr'), 'rb').readlines()

    aw = open(os.path.join('wmt14_3', str(max_length)+'_train.de'), 'wb')
    bw = open(os.path.join('wmt14_3', str(max_length)+'_train.en'), 'wb')
    cw = open(os.path.join('wmt14_3', str(max_length)+'_train.fr'), 'wb')

    num = len(a)
    print('num: ', num)
    i = 0
    for idx in range(num):
        if len(a[idx].split())<max_length and len(b[idx].split())<max_length and len(c[idx].split())<max_length:
            i = i + 1
            aw.write(a[idx])
            bw.write(b[idx])
            cw.write(c[idx])
    aw.close()
    bw.close()
    cw.close()
    print('final num: ', i)


if __name__ == '__main__':
    main()