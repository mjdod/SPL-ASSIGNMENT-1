all: bin/server

bin/server: bin/Main.o bin/Model.o bin/Server.o
	@echo 'Linking'
	g++ -o bin/server bin/Main.o bin/Model.o bin/Server.o
bin/Server.o: src/Server.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Server.cpp -o bin/Server.o
bin/Main.o: src/Main.cpp 
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Main.cpp -o bin/Main.o
bin/Model.o: src/Model.cpp 
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Model.cpp -o bin/Model.o
.PHONY:clean
clean:
	rm -f bin/*
