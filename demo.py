import re
import numpy as np
import matplotlib.pyplot as plt
# Initial Permutation
IP = [58, 50, 42, 34, 26, 18, 10, 2,
			60, 52, 44, 36, 28, 20, 12, 4,
			62, 54, 46, 38, 30, 22, 14, 6,
			64, 56, 48, 40, 32, 24, 16, 8,
			57, 49, 41, 33, 25, 17, 9,  1,
			59, 51, 43, 35, 27, 19, 11, 3,
			61, 53, 45, 37, 29, 21, 13, 5,
			63, 55, 47, 39, 31, 23, 15, 7]

# Permutation Choice One, 64bits -> 56bits            
PC1 = [ 57, 49, 41, 33, 25, 17,  9,  1, 58, 50, 42, 34, 26, 18,
    10,  2, 59, 51, 43, 35, 27, 19, 11,  3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,  7, 62, 54, 46, 38, 30, 22,
    14,  6, 61, 53, 45, 37, 29, 21, 13,  5, 28, 20, 12,  4
]

# Permutation Choice Two, 56bits -> 48bits
PC2 = [ 14, 17, 11, 24,  1,  5,  3, 28, 15,  6, 21, 10,
    23, 19, 12,  4, 26,  8, 16,  7, 27, 20, 13,  2,
    41, 52, 31, 37, 47, 55, 30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32
]

# Schedule of Left Shift
SLS = [ 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

# Expansion 32bits -> 48bits
E = [32,  1,  2,  3,  4,  5,  4,  5,  6,  7,  8,  9,
    8,  9, 10, 11, 12, 13, 12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21, 20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32,  1
]

# S-box
S_BOX = [ [ [14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7],
		[0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8],  
		[4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0],
		[15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13] ],
	[ [15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10],  
	[3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5], 
	[0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15], 
	[13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9] ],
	[ [10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8],  
	[13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1],
	[13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7],  
	[1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12] ],
	[ [7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15],  
	[13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9],  
	[10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4],  
	[3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14] ],
	[[2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9],  
		[14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6],  
		[4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14],  
		[11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3] ],
	[ [12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11],  
		[10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8],  
		[9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6],  
		[4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13] ],
	[[4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1],  
		[13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6],  
		[1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2],  
		[6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12] ], 
	[ [13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7],  
		[1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2],  
		[7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8],  
		[2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11] ]]

# Permutation(P) box
P = [16,  7, 20, 21,
		   29, 12, 28, 17,
		    1, 15, 23, 26,
		    5, 18, 31, 10,
		    2,  8, 24, 14,
		   32, 27,  3,  9,
		   19, 13, 30,  6,
		   22, 11,  4, 25 ]

# Final Permutation
FP = [40, 8, 48, 16, 56, 24, 64, 32,
			  39, 7, 47, 15, 55, 23, 63, 31,
			  38, 6, 46, 14, 54, 22, 62, 30,
			  37, 5, 45, 13, 53, 21, 61, 29,
			  36, 4, 44, 12, 52, 20, 60, 28,
			  35, 3, 43, 11, 51, 19, 59, 27,
			  34, 2, 42, 10, 50, 18, 58, 26,
			  33, 1, 41,  9, 49, 17, 57, 25]

# initial IP change process
def IP_transform(data):
	temp = [0 for _ in range(64)]
	for i in range(64):
		temp[i] = data[i]
	for i in range(0,64):
		data[i] = temp[IP[i] - 1] 

# inverse IP^(-1) FP change process
def FP_transform(data):
	temp = [0 for _ in range(64)]
	for i in range(64):
		temp[i] = data[i]
	for i in range(64):
		data[i] = temp[FP[i] - 1]

# PC1 transform of securit key
def PC1_replace(key, key_56):
	for i in range(56):
		key_56[i] = key[PC1[i] - 1]

# PC2 transform of securit key
def PC2_replace(key_56, key_48):
	for i in range(48):
		key_48[i] = key_56[PC2[i] - 1]

# Looping Left move process
def SLS_leftshit(c, d, m):
	if(m == 1 or m == 2 or m == 9 or m == 16):
		temp_c = c[0]
		temp_d = d[0]
		for i in range(27):
			c[i] = c[i+1]
			d[i] = d[i+1]
		c[27] = temp_c
		d[27] = temp_d
	else:
		temp_c0 = c[0]
		temp_d0 = d[0]
		temp_c1 = c[1]
		temp_d1 = d[1]
		for i in range(26):
			c[i] = c[i+2]
			d[i] = d[i+2]
		c[26] = temp_c0
		d[26] = temp_d0
		c[27] = temp_c1
		d[27] = temp_d1

# E box change process
def E_extend(data_32, data_48):
	for i in range(48):
		data_48[i] = data_32[E[i] - 1]

# P box change process
def P_replace(data):
	temp = [0 for _ in range(32)]
	for i in range(32):
		temp[i] = data[i]
	for i in range(32):
		data[i] = temp[P[i] - 1]

# S box change process
def S_replace(data, data_out):
	for i in range(8):
		temp1 = i * 6
		temp2 = i * 4
		row = (data[temp1] * 2) + data[temp1 + 5]
		col = data[temp1 + 1] * 8 + data[temp1 + 2] * 4 + data[temp1 + 3] * 2 + data[temp1 + 4]
		out = S_BOX[i][row][col]
		data_out[temp2] = out % 2
		out /= 2
		data_out[temp2 + 1] = out % 2
		out /= 2
		data_out[temp2 + 2] = out % 2
		out /= 2
		data_out[temp2 + 3] = out % 2

# string to the binary
def stringtobinary(s):
    b = ""
    for i in s:  
        asc = bin(ord(i))[2:] 
        for j in range(0,8-len(asc)):
            asc = '0'+ asc
        b += asc
    return b
# binary to string
def binarytostring(b):
    s = ""
    tmp = re.findall(r'.{8}',b)  #每8位表示一个字符
    for i in tmp:
        s += chr(int(i,2))  #base参数的意思，将该字符串视作2进制转化为10进制
    return s

key = [1 for _ in range(64)]
keyPC1 = [0 for _ in range(56)]
keyCD= [0 for _ in range(56)]
C= [0 for _ in range(28)]
D= [0 for _ in range(28)]
K = []
for i in range(16):
	K.append([0]*48)


def Generate_key():
	SecurityKey = 'CISC7015'
	key = stringtobinary(SecurityKey)
	# j = 0
	# for i in range(8):
	# 	a = [0 for _ in range(8)]
	# 	m = SecurityKey[i]
	# 	while (m == 0):
	# 		a[j] = m % 2
	# 		m = m /2
	# 		j += 1
	# 		for j in range(8):
	# 			key[(i * 8) + j] = a[7 - j]
	# key = []
	# key = encode_char_to_bin(SecurityKey)
	# key = list(map(int, key))
	print('Security Key Binary Reprensent is')
	print(key)
	print('###### Generating Subkey ######')
	# print("".join(str(i) for i in key))
	key = list(map(int, key))
	PC1_replace(key, keyPC1)
	for i in range(28):
		C[i] = keyPC1[i]
		D[i] = keyPC1[i+28]
	for t in range(16):
		SLS_leftshit(C, D, t+1)
		for i in range(28):
			keyCD[i] = C[i]
			keyCD[i + 28] = D[i]
		PC2_replace(keyCD, K[t])
	for i in range(16):
		print('SubKey',i + 1, 'is')
		tmp = []
		for j in range(48):
			tmp.append(K[i][j])
		print("".join(str(i) for i in tmp))

def encryptanddecrypt(plaintext):
	text = [0 for _ in range(64)]
	L0 = [0 for _ in range(32)]
	R0 = [0 for _ in range(32)]
	data_e = [0 for _ in range(48)]
	data_s = [0 for _ in range(32)]
	data_p = [0 for _ in range(32)]
	temp = [0 for _ in range(32)]
	plaintext_out = []
	encrypted_out = []
	Generate_key()
	count = len(plaintext)
	k = count % 64
	if (k == 0):
		num = count // 64
	else:
		num = (count - k) // 64 + 1
	for r in range(0,num):
		cnt = 0
		for i in range(64):
			tmpi = i + 64 * r
			if(tmpi > count -1):
				cnt += 1
				text[i] = '0'
			else:
				text[i] = plaintext[tmpi]
		print('cnt is ',cnt)
		# tmp_text = "".join(text)
		# bin_text = stringtobinary(tmp_text)
		# bin_text = bin_text.ljust(64,'0')
		print('###The',r+1,'time encryption and decryption###')
		data = list(map(int, text))
		IP_transform(data)
		for j in range(32):
			L0[j] = data[j]
			R0[j] = data[j + 32]
		for i in range(16):
			for j in range(32):
				temp[j] = R0[j]
			E_extend(R0, data_e)
			for j in range(48):
				data_e[j] = int(data_e[j]) ^ int(K[i][j])
				S_replace(data_e, data_s)
				P_replace(data_s)
			for j in range(32):
				R0[j] = int(L0[j]) ^ int(data_s[j])
			for j in range(32):
				L0[j] = temp[j]
		for i in range(32):
			data[i] = R0[i]
			data[i + 32] = L0[i]
		FP_transform(data)
		data_new = [str(x) for x in data]
		print('Ciphertext encrypt binary is')
		tmpci = "".join(data_new)
		encrypted_out.append(tmpci)
		print(tmpci)	
		
		# DECRYPTION
		data = [int(x) for x in data]
		IP_transform(data) # initial permutation
		for j in range(32): # the front 32-bits data of the plaintext
			L0[j] = data[j]
			R0[j] = data[j + 32]
		for i in range(16): # 16 rounds iterations
			for j in range(32):
				temp[j] = R0[j]
			E_extend(R0, data_e) # E extend
			for j in range(48):
				data_e[j] = int(data_e[j]) ^ int(K[15 - i][j])
			S_replace(data_e, data_s) # S box changed
			P_replace(data_s) # P box changed
			for j in range(32):
				R0[j] = int(L0[j]) ^ int(data_s[j])
			for j in range(32):
				L0[j] = temp[j]
		for i in range(32):
			data[i] = R0[i]
			data[i + 32] = L0[i]
		FP_transform(data)
		print('The Decrypted plaintext binary is')
		data_new2 = [str(x) for x in data]
		if cnt != 0:
			data_new2 = data_new2[:-cnt]
		data_str = "".join(data_new2)
		print(data_str)
		plaintext_out.append(data_str)
	print('\n')
	res = "".join(plaintext_out)
	res_encrypted = "".join(encrypted_out)
	return res, res_encrypted
				

if __name__=="__main__":
	print('1. Encrypted the BIN data')
	print('2. Encrypted the DEC data')
	print('3. Encrypted the HEX data')
	print('4. Encrypted the STRING data')
	print('5. Encrypted the IMAGE data')
	x = int(input('please input an integer to decide what data format you wanna ciper: '))
	if x == 1:
		plaintext = input('please input the BIN data with only 0 and 1: ')
		finalbin = encryptanddecrypt(plaintext)
		print('The final encrypted plaintext is(BIN format):')
		print(finalbin[1])
		print('The final decryted plaintext is(BIN format):')
		print(finalbin[0])
	elif x == 2:
		plaintext = input('please input the DEC data with 0-9: ')
		plaintext = bin(int(plaintext))
		plaintext = plaintext[2:]
		finalbin = encryptanddecrypt(plaintext)
		finaloct = int(finalbin[0],2)
		finaloct_en = int(finalbin[1],2)
		print('The final encrypted plaintext is(OCT format):')
		print(finaloct_en)
		print('The final decryted plaintext is(OCT format):')
		print(finaloct)
	elif x == 3:
		plaintext = input('pleaze input the HEX data with 0-9 and A-F, a-f: ')
		plaintext = bin(int(plaintext,16))
		plaintext = plaintext[2:]
		finalbin = encryptanddecrypt(plaintext)
		finalhex = hex(int(finalbin[0],2))
		finalhex_en = hex(int(finalbin[1],2))
		print('The final encrypted plaintext is(HEX format):')
		print(finalhex_en)
		print('The final decryted plaintext is(HEX format):')
		print(finalhex)
	elif x == 4:
		plaintext = input('please input the string data with char a-z, A-Z, 0-9, and other symols: ')
		plaintext = stringtobinary(plaintext)
		finalbin = encryptanddecrypt(plaintext)
		print('The final encrypted plaintext is(STRING format):')
		print(binarytostring(finalbin[1]))
		print('The final decryted plaintext is(STRING format):')
		print(binarytostring(finalbin[0]))
	elif x == 5:
		plaintext = input('please input the image file location: ')	
		img = plt.imread(plaintext)
		plainimage = np.array(img)
		P2 = plainimage.copy
		print(plainimage.shape)

		text = [0 for _ in range(64)]
		L0 = [0 for _ in range(32)]
		R0 = [0 for _ in range(32)]
		data_e = [0 for _ in range(48)]
		data_s = [0 for _ in range(32)]
		data_p = [0 for _ in range(32)]
		temp = [0 for _ in range(32)]
		plaintext_out = []
		encrypted_out = []
		encrypted_image = np.zeros((256,256))
		decrypted_image = np.zeros((256,256))
		Generate_key()

		for i in range(256):
			for j in range(256):
				plaintext = bin(int(plainimage[i,j]))
				plaintext = plaintext[2:]
				cnt = 0
				count = len(plaintext)
				for i in range(64):
					if(i > count -1):
						cnt += 1
						text[i] = '0'
					else:
						text[i] = plaintext[i]
				data = list(map(int, text))
				IP_transform(data)
				for j in range(32):
					L0[j] = data[j]
					R0[j] = data[j + 32]
				for i in range(16):
					for j in range(32):
						temp[j] = R0[j]
					E_extend(R0, data_e)
					for j in range(48):
						data_e[j] = int(data_e[j]) ^ int(K[i][j])
						S_replace(data_e, data_s)
						P_replace(data_s)
					for j in range(32):
						R0[j] = int(L0[j]) ^ int(data_s[j])
					for j in range(32):
						L0[j] = temp[j]
				for i in range(32):
					data[i] = R0[i]
					data[i + 32] = L0[i]
				FP_transform(data)
				data_new = [str(x) for x in data]
				tmpci = "".join(data_new)
				en_image = int(tmpci,2)
				encrypted_image[i,j] = en_image
				
				
				# DECRYPTION
				data = [int(x) for x in data]
				IP_transform(data) # initial permutation
				for j in range(32): # the front 32-bits data of the plaintext
					L0[j] = data[j]
					R0[j] = data[j + 32]
				for i in range(16): # 16 rounds iterations
					for j in range(32):
						temp[j] = R0[j]
					E_extend(R0, data_e) # E extend
					for j in range(48):
						data_e[j] = int(data_e[j]) ^ int(K[15 - i][j])
					S_replace(data_e, data_s) # S box changed
					P_replace(data_s) # P box changed
					for j in range(32):
						R0[j] = int(L0[j]) ^ int(data_s[j])
					for j in range(32):
						L0[j] = temp[j]
				for i in range(32):
					data[i] = R0[i]
					data[i + 32] = L0[i]
				FP_transform(data)
				data_new2 = [str(x) for x in data]
				if cnt != 0:
					data_new2 = data_new2[:-cnt]
				data_str = "".join(data_new2)
				data_str = int(data_str,2)
				decrypted_image[i,j] = data_str
		plt.figure()
		plt.imshow(plainimage)
		plt.imshow(encrypted_image)
		plt.imshow(decrypted_image)
		plt.show()
