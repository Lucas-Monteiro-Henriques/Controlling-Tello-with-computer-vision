Os fiscalizadores de aeronaves comerciais desempenham um papel indispensável na pista, servindo como ponte visual entre os pilotos e a tripulação de terra. Munidos com varinhas de sinalização – faróis portáteis iluminados – eles comunicam instruções vitais aos pilotos, desde desacelerar e virar até parar ou desligar os motores. Essa sinalização visual, caracterizada pela sua intuitividade e redundância, garante clareza e minimiza erros. Esse cenário resume o potencial de aproveitar os movimentos do corpo humano como entradas de comando dentro dos sistemas, estabelecendo as bases para a avançada Interação Humano-Robô (HRI). O HRI, como domínio, investiga as complexidades da incorporação de humanos no circuito de controle, influenciando assim o comportamento robótico.

No contexto da aviação e da robótica, o uso de gestos corporais para controle de máquinas tem ganhado cada vez mais espaço. Um exemplo prático disso é a implementação de visão computacional para controlar um drone DJI Tello por meio de reconhecimento de poses, utilizando a biblioteca MediaPipe e OpenCV. O código fornecido demonstra como essa abordagem pode ser aplicada na prática.

O script inicializa o drone e ativa sua câmera, permitindo a captação de imagens em tempo real. Em seguida, utiliza a biblioteca MediaPipe para detectar posições corporais e OpenCV para processar as imagens. Dependendo da pose detectada, o drone recebe comandos para avançar, recuar, subir, pousar ou manter-se estável. O código também incorpora um controle PID para ajuste fino do movimento do drone com base na detecção de rostos, garantindo maior precisão no rastreamento.

O funcionamento do código baseia-se em três etapas principais:

Aquisição e processamento de imagens: A câmera do drone transmite os frames para processamento. O MediaPipe é utilizado para detectar landmarks corporais e OpenCV para realizar detecção de rosto.

Análise dos gestos e rostos: As poses do usuário são interpretadas e mapeadas para comandos específicos. Por exemplo, levantar um dos braços pode fazer o drone avançar ou recuar.

Envio de comandos ao drone: Após a interpretação dos movimentos, o drone recebe comandos via send_rc_control(), ajustando sua posição no espaço.

Esse método não apenas aprimora a interação entre humanos e máquinas, mas também abre novas possibilidades para o controle de robôs em ambientes dinâmicos, como aeroportos e fábricas. Ao substituir controladores convencionais por gestos intuitivos, a comunicação torna-se mais eficiente e acessível, reduzindo barreiras tecnológicas e aprimorando a segurança nas operações aéreas e industriais.


Link para o video do TEllo no youtube: https://youtu.be/TFsnT5g8JX4?si=1TKhHvfnrXGtRzSH
