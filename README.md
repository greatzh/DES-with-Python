# DES-with-Python
## using des algorithm encrypt the data including the binary, decimal, hexadecimal, string, and image(with some bugs now)
In this implementation, we use the ascii code and zero-padding.
The DES algorithm is that we input a 64-bit plaintext at first, then do the initial permutation. After that, 16 rounds of identical 
processing stages under the control of security key. Next, the swapping of left and right, the final 
permutation, which is IP’s inversing version. Finally, the 64-bit ciphertext output.
