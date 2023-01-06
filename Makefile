FILE=main.cu
TARGET=main
TEACHER=karwowskij

all: ${TARGET}

${TARGET}: ${FILE}
	nvcc ${FILE} -o ${TARGET}

# # wysyłanie plików
# send: z zcheck s scheck

# # pakowanie plików
# z:
# 	tar -cjf zukowskal.tar.bz2 ${FILES} Makefile

# # sprawdzenie, czy wszystkie pliki zostały zapakowane
# zcheck:
# 	tar -tjf zukowskal.tar.bz2

# # wysyłanie plików
# s:
# 	cp zukowskal.tar.bz2 /home2/samba/${TEACHER}/unix/

# # sprawdzenie, czy plik został wysłany
# scheck:
# 	ls -l /home2/samba/${TEACHER}/unix/zukowskal.tar.bz2

.Phony: clean all #z zcheck s scheck send

clean: 
	rm ${TARGET}
	
#zukowskal.tar.bz2
