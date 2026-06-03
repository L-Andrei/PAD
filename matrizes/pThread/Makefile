# Variáveis do Compilador
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20 -O3 -mavx2 -mfma -pthread
LDFLAGS = -lgtest -lgtest_main -pthread

# Nomes dos arquivos
TARGET = programa_matriz
TEST_TARGET = executar_testes
SRC = matrix.hpp
TEST_SRC = test.cpp

# Regra padrão (compila tudo)
all: $(TARGET)

# Compila o programa principal (caso você tenha um main.cpp)
$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o $(TARGET)

# Compila e vincula os testes com o Google Test
$(TEST_TARGET): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $(TEST_SRC) -o $(TEST_TARGET) $(LDFLAGS)

# Atalho para rodar os testes
test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Limpa os binários e arquivos temporários
clean:
	rm -f $(TARGET) $(TEST_TARGET) *.o

# Ajuda
help:
	@echo "Comandos disponíveis:"
	@echo "  make all     - Compila o programa principal"
	@echo "  make test    - Compila e executa os testes unitários (GTest)"
	@echo "  make clean   - Remove os arquivos executáveis"