def Levenshtein_Distance(str1, str2):
    matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1 
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)

    return matrix[len(str1)][len(str2)]

def cal_cer_ed(path_ours, tail='_rec'):
    path_gt='./GT/'
    N=66
    cer1=[]
    cer2=[]
    ed1=[]
    ed2=[]
    check=[0 for _ in range(N+1)]
    lis=[1,2,3,4,5,6,7,9,10,21,22,23,24,27,30,31,32,36,38,40,41,44,45,46,47,48,50,51,52,53]  # DocTr (Setting 1)
    # lis=[1,9,10,12,19,20,21,22,23,24,30,31,32,34,35,36,37,38,39,40,44,45,46,47,49] # DewarpNet (Setting 2)
    for i in range(1,N):
        if i not in lis:
            continue
        gt=Image.open(path_gt+str(i)+'.png')
        img1=Image.open(path_ours+str(i)+'_1' + tail)
        img2=Image.open(path_ours+str(i)+'_2' + tail)
        content_gt=pytesseract.image_to_string(gt)
        content1=pytesseract.image_to_string(img1)
        content2=pytesseract.image_to_string(img2)
        l1=Levenshtein_Distance(content_gt,content1)
        l2=Levenshtein_Distance(content_gt,content2)
        ed1.append(l1)
        ed2.append(l2)
        cer1.append(l1/len(content_gt))
        cer2.append(l2/len(content_gt))
        check[i]=cer1[-1]
    print('CER: ', (np.mean(cer1)+np.mean(cer2)) / 2.)
    print('ED:  ', (np.mean(ed1)+np.mean(ed2)) / 2.)

def evalu(path_ours, tail):
    cal_cer_ed(path_ours, tail)
