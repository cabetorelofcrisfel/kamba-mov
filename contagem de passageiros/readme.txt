Perfeito, vou estruturar a lógica do programa de forma clara para que possas guardar no teu README. Assim tens um guia de como o sistema deve funcionar do início ao fim:

---

## 📌 Lógica do Sistema de Monitorização de Passageiros

### 1. **Detecção de Entrada**
- A câmera detecta quando uma pessoa entra no táxi.  
- O sistema captura o rosto e gera um **ID temporário** (embedding facial ou hash).  
- Esse ID é usado apenas para diferenciar passageiros, sem necessidade de guardar dados pessoais.

### 2. **Validação da Viagem**
- O passageiro só é considerado válido se o carro **iniciar movimento** após a entrada.  
- Se o passageiro sair antes do carro andar, o ID é descartado (não conta como viagem).  
- Se o passageiro entrar e sair várias vezes antes da viagem começar, continua a ser ignorado.

### 3. **Registo de Ponto de Entrada**
- Quando o carro começa a deslocar-se, o sistema associa o ID do passageiro ao **ponto de entrada (GPS)**.  
- Esse registo marca o início da viagem para aquele passageiro.

### 4. **Detecção de Saída**
- Quando o passageiro sai, a câmera confirma a saída e o sistema regista o **ponto de saída (GPS)**.  
- O ID é encerrado e a viagem é considerada concluída.

### 5. **Cálculo do Valor da Viagem**
- O sistema calcula a distância entre o ponto de entrada e saída.  
- Aplica a tarifa do táxi (ex.: preço por km + taxa fixa).  
- Associa o valor ao passageiro identificado.

### 6. **Gestão de Casos Especiais**
- **Passageiro que entra e sai sem viagem** → descartado.  
- **Passageiro que entra para ajudar outro** → descartado.  
- **Passageiro que entrou no carro errado** → descartado se sair antes do carro andar.  
- **Passageiro que faz viagem parcial** → valor calculado apenas pela distância percorrida.

### 7. **Relatório Diário**
- No fim do dia, o sistema gera um relatório com:  
  - Número total de passageiros transportados.  
  - Lista de viagens válidas (entrada/saída + valor).  
  - Receita total do dia.

---

## 🔄 Fluxo Simplificado em Pseudocódigo

```pseudo
quando pessoa entra:
    capturar rosto -> gerar ID
    marcar estado = "aguardando viagem"

quando carro começa a andar:
    para cada ID em estado "aguardando viagem":
        marcar estado = "em viagem"
        registar ponto de entrada (GPS)

quando pessoa sai:
    se estado == "em viagem":
        registar ponto de saída (GPS)
        calcular valor da viagem
        adicionar ao relatório diário
        remover ID
    senão:
        descartar ID (entrada inválida)

no fim do dia:
    gerar relatório com total de passageiros e receita
```

---

👉 Esta lógica resolve os problemas que mencionaste: evita duplicações, descarta entradas inválidas e calcula corretamente o valor de cada viagem.  

Queres que eu te ajude a transformar este pseudocódigo em um **fluxo visual (diagrama)** para deixar o README ainda mais claro?