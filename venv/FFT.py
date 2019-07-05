'''
canny_open = cv2.Canny(open_template, 300, 150)
canny_close = cv2.Canny(close_template, 300, 150)
fft_open = np.fft.fftshift(np.fft.fft2(canny_open))
fft_open_norm = np.abs(fft_open)
fft_open_norm = np.uint8(fft_open_norm/np.max(fft_open_norm)*255)
print(np.max(fft_open_norm))
#fft_open_norm = cv2.normalize(np.abs(fft_open), np.abs(fft_open_norm),0,255, norm_type=cv2.NORM_MINMAX)
print(fft_open_norm)
cv2.imshow('open_fft', fft_open_norm)

fft_close = np.fft.fftshift(np.fft.fft2(canny_close))
fft_close_norm = np.abs(fft_close)
fft_close_norm = np.uint8(fft_close_norm/np.max(fft_close_norm)*255)
print(np.max(fft_close_norm))
#fft_open_norm = cv2.normalize(np.abs(fft_open), np.abs(fft_open_norm),0,255, norm_type=cv2.NORM_MINMAX)
print(fft_close_norm)
cv2.imshow('close_fft', fft_close_norm)

cv2.waitKey()
cv2.destroyAllWindows()
'''