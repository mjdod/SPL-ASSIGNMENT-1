all: bin/server

bin/server: bin/Main.o bin/Model.o bin/Server.o bin/GPU.o bin/MemoryBlock.o bin/Request.o bin/SchedulingPolicy.o
	@echo 'Linking'
	g++ -o bin/server bin/Main.o bin/Model.o bin/Server.o bin/GPU.o bin/MemoryBlock.o bin/Request.o bin/SchedulingPolicy.o

bin/Server.o: src/Server.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Server.cpp -o bin/Server.o

bin/Main.o: src/Main.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Main.cpp -o bin/Main.o

bin/Model.o: src/Model.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Model.cpp -o bin/Model.o

bin/GPU.o: src/GPU.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/GPU.cpp -o bin/GPU.o

bin/MemoryBlock.o: src/MemoryBlock.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/MemoryBlock.cpp -o bin/MemoryBlock.o

bin/Request.o: src/Request.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Request.cpp -o bin/Request.o

bin/SchedulingPolicy.o: src/SchedulingPolicy.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/SchedulingPolicy.cpp -o bin/SchedulingPolicy.o

.PHONY: clean
clean:
	rm -f bin/*