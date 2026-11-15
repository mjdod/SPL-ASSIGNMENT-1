all: bin/Server

bin/Server: bin/Main.o bin/Model.o 
	@echo 'Linking'
	g++ -o bin/Server bin/Main.o bin/Model.o
bin/Main.o: src/Main.cpp 
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Main.cpp -o bin/Main.o
bin/Model.o: src/Model.cpp 
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Model.cpp -o bin/Model.o
.phony:clean
clean:
	@echo 'cleaning'
	rm -f bin/*.o bin/main
