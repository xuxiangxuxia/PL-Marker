import json
import os
import sys
import time
import numpy as np
import paramiko
import json
from nltk.tokenize import word_tokenize
import spacy
import select
from PyQt5 import QtGui, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QRadioButton, QButtonGroup, \
    QMessageBox, QMainWindow, QVBoxLayout, QTextEdit, QGridLayout, QComboBox
from PyQt5.QtGui import QIcon, QPalette, QBrush, QPixmap, QTextBlockFormat, QTextCursor, QColor
from PyQt5.QtCore import pyqtSlot, QProcess, QTimer, QThread, pyqtSignal
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SSHThread(QThread):
    output_signal = pyqtSignal(str)
    def __init__(self, hostname, port, username, password, command):
        super().__init__()
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.command = command
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def run(self):
        try:
            self.ssh.connect(self.hostname, self.port, self.username, self.password)

            channel = self.ssh.get_transport().open_session()
            channel.get_pty()  # 获取伪终端
            # 执行命令，并将进度信息写入文件
            channel.exec_command(self.command)
            output_data = ''
            while True:
                if channel.exit_status_ready():  # 命令执行完毕
                    break
                rl, wl, xl = select.select([channel], [], [], 0.0)
                if len(rl) > 0:
                    # 读取输出数据并发送信号到主线程更新界面
                    data = channel.recv(1024).decode('utf-8')
                    output_data += data
                    self.output_signal.emit(data)
                    #self.update_output_text(output_data)
                    time.sleep(0.1)

            self.ssh.close()
        except paramiko.AuthenticationException as e:
            self.output_signal.emit(str(e))
        except Exception as e:
            self.output_signal.emit(str(e))
        finally:
            self.ssh.close()


class App(QWidget):
    commbase_ner = 'CUDA_VISIBLE_DEVICES=1  python3  run_acener.py   \
    --model_name_or_path  bert_models/scibert_scivocab_uncased  --do_lower_case  \
    --data_dir scierc  \
    --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 263  --max_pair_length 256  --max_mention_ori_length 8    \
    --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed 42  --onedropout  --lminit  \
    --train_file train.json --dev_file dev.json --test_file test.json  \
     --overwrite_output_dir  --output_results'
    commbase_re = 'CUDA_VISIBLE_DEVICES=1  python3  run_re.py  \
    --model_name_or_path  bert_models/scibert_scivocab_uncased  --do_lower_case  \
    --data_dir scierc  \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --max_pair_length 16  --save_steps 904  \
    --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
    --fp16   \
    --test_file sciner_models/sciner-scibert/ent_pred_test.json  \
    --use_ner_results '
    commbase_prompt_re = 'CUDA_VISIBLE_DEVICES=1  python3  run_re.py  \
        --model_name_or_path  bert_models/scibert_scivocab_uncased  --do_lower_case  \
        --data_dir scierc  \
        --learning_rate 2e-5  --num_train_epochs 30  --per_gpu_train_batch_size  16  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
        --max_seq_length 256  --max_pair_length 1  --save_steps 1755  \
        --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
        --fp16   \
        --test_file sciner_models/sciner-scibert/ent_pred_test.json  \
        --use_ner_results '
    commbase_ner_prompt = 'CUDA_VISIBLE_DEVICES=1  python3  run_acener.py   \
        --model_name_or_path  bert_models/scibert_scivocab_uncased  --do_lower_case  \
        --data_dir scierc  \
        --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  24  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
        --max_seq_length 512  --save_steps 12969  --max_pair_length 1  --max_mention_ori_length 8    \
        --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
        --fp16  --seed 42  --onedropout  --lminit  \
        --train_file train.json --dev_file dev.json --test_file test.json  \
         --overwrite_output_dir  --output_results'
    commbase_re_levitated = 'CUDA_VISIBLE_DEVICES=1  python3  run_levitatedpair.py  \
    --model_name_or_path  bert_models/scibert_scivocab_uncased  --do_lower_case  --data_dir scierc \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size 8  --per_gpu_eval_batch_size  16  \
    --gradient_accumulation_steps  1  --max_seq_length  256  --max_pair_length 16  --save_steps 294  \
    --do_eval  --evaluate_during_training  --eval_all_checkpoints  --eval_logsoftmax  --fp16   --seed 42 \
    --test_file sciner_models/sciner-scibert/ent_pred_test.json  --use_ner_results  \
    --overwrite_output_dir'
    commbase_robert_ner = 'CUDA_VISIBLE_DEVICES=1  python3  run_acener.py   \
       --model_name_or_path  bert_models/roberta-base  --do_lower_case  \
       --data_dir scierc  \
       --learning_rate 1e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
       --max_seq_length 512  --save_steps 263  --max_pair_length 256  --max_mention_ori_length 8    \
       --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
       --fp16  --seed 42  --onedropout  --lminit  \
       --train_file train.json --dev_file dev.json --test_file test.json  \
        --overwrite_output_dir  --output_results'
    commbase_albert_ner = 'CUDA_VISIBLE_DEVICES=1  python3  run_acener.py   \
           --model_name_or_path  bert_models/albert-base-v2  --do_lower_case  \
           --data_dir scierc  \
           --learning_rate 1e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
           --max_seq_length 512  --save_steps 263  --max_pair_length 256  --max_mention_ori_length 8    \
           --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
           --fp16  --seed 42  --onedropout  --lminit  \
           --train_file train.json --dev_file dev.json --test_file test.json  \
            --overwrite_output_dir  --output_results'
    commbase_albert_re = 'CUDA_VISIBLE_DEVICES=1  python3  run_re.py  \
            --model_name_or_path  bert_models/albert-base-v2  --do_lower_case  \
            --data_dir scierc  \
            --learning_rate 1e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
            --max_seq_length 256  --max_pair_length 16  --save_steps 880  \
            --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
            --fp16   \
            --test_file sciner_models/sciner-scibert/ent_pred_test.json  \
            --use_ner_results '


    model = { 're':'  --model_type bertsub'+'  --output_dir scire_models/scire-scibert',
              'ner_albert':'  --model_type albertspanmarker'+'  --output_dir sciner_models/sciner-scialbert',
              're_albert':'  --model_type albertsub' + '  --output_dir scire_models/scire-scialbert',
              're_prompt':'  --model_type bertsub' + '  --output_dir scire_models/scire-scibert-prompt',
              're_albert_1': '  --model_type albertsub_1' + '  --output_dir scire_models/scire-scialbert_1',
              're_typemarker': '  --model_type bertsub' + '  --output_dir scire_models/scire-scibert-typemarker' + '  --use_typemarker',
              're_1':'  --model_type bertsub_1' + '  --output_dir scire_models/scire-scibert_1',
              're_1_span': '  --model_type bertsub_1_span' + '  --output_dir scire_models/scire-scibert_1_span',
              're_1_span_typemarker': '  --model_type bertsub_1_span' + '  --output_dir scire_models/scire-scibert_1_span_typemarker' + '  --use_typemarker',
              're_1_typemarker': '  --model_type bertsub_1' + '  --output_dir  scire_models/scire-scibert_1-typemarker' + '  --use_typemarker',
              're_levitated': '  --model_type bert' + '  --output_dir  scire_models/scire-scibert-levitatedpair',
              're_bilevitated': '  --model_type bert_blinear' + '  --output_dir  scire_models/scire-scibert-bilevitatedpair',
              're_bilevitated_1':  '  --model_type bert_blinear_1' + '  --output_dir  scire_models/scire-scibert-bilevitatedpair-1',
              're_bilevitated_typemarker': '  --model_type bert_blinear_1' + '  --output_dir  scire_models/scire-scibert-bilevitatedpair-typemarker' + '  --use_typemarker',
              'ner': '  --model_type bertspanmarker' + '  --output_dir sciner_models/sciner-scibert',
              'ner_prompt': '  --model_type bertspanmarker' + '  --output_dir sciner_models/sciner-scibert-prompt',
              'ner_onlyentity':  '  --model_type bertspanmarker_onlyentity ' + '  --output_dir sciner_models/sciner-onlyentity-scibert',
              'ner_onlymarker':  '  --model_type bertspanmarker_onlymarker ' + '  --output_dir sciner_models/sciner-onlymarker-scibert',
              'ner_bertspanmarkerbi':  '  --model_type bertspanmarkerbi ' + '  --output_dir sciner_models/sciner-scibert-blinear',
              'ner_bertspanmarkerbi_spanmarke': '  --model_type bertspanmarkerbi ' + '  --output_dir sciner_models/sciner-scibert-blinear' + '  --use_typemarker',
              'ner_roberta': '  --model_type robertaspanmarker' + '  --output_dir sciner_models/sciner-sciroberta',
              'ner_roberta_prompt': '  --model_type robertaspanmarker' + '  --output_dir sciner_models/sciner-sciroberta-prompt',

              }
    comms={'re_evaluate':commbase_re+'  --model_type bertsub'+ '  --output_dir scire_models/scire-scibert',
           're_train': commbase_re + '  --model_type bertsub'+ '  --do_train'+'  --output_dir scire_models/scire-scibert',

           're_prompt_evaluate': commbase_prompt_re + '  --model_type bertsub' + '  --output_dir scire_models/scire-scibert-prompt',
           're_prompt_train': commbase_prompt_re + '  --model_type bertsub' + '  --do_train' + '  --output_dir scire_models/scire-scibert-prompt',

           'ner_albert_evaluate': commbase_albert_ner + '  --model_type albertspanmarker' + '  --output_dir sciner_models/sciner-scialbert',
           'ner_albert_train': commbase_albert_ner + '  --model_type albertspanmarker' + '  --do_train' + '  --output_dir sciner_models/sciner-scialbert',

           're_albert_evaluate': commbase_albert_re + '  --model_type albertsub' + '  --output_dir scire_models/scire-scialbert',
           're_albert_train': commbase_albert_re + '  --model_type albertsub' + '  --do_train' + '  --output_dir scire_models/scire-scialbert',

           're_albert_1_evaluate': commbase_albert_re + '  --model_type albertsub_1' + '  --output_dir scire_models/scire-scialbert_1',
           're_albert_1_train': commbase_albert_re + '  --model_type albertsub_1' + '  --do_train' + '  --output_dir scire_models/scire-scialbert_1',

           're_typemarker_train':commbase_re + '  --model_type bertsub'+ '  --do_train'+'  --output_dir scire_models/scire-scibert-typemarker'+'  --use_typemarker',
           're_typemarker_evaluate': commbase_re + '  --model_type bertsub'+ '  --output_dir scire_models/scire-scibert-typemarker' + '  --use_typemarker',

           're_1_train': commbase_re + '  --model_type bertsub_1'+ '  --do_train'+'  --output_dir scire_models/scire-scibert_1',
           're_1_evaluate': commbase_re + '  --model_type bertsub_1'+'  --output_dir scire_models/scire-scibert_1',

           're_1_span_train': commbase_re + '  --model_type bertsub_1_span' + '  --do_train' + '  --output_dir scire_models/scire-scibert_1_span',
           're_1_span_evaluate': commbase_re + '  --model_type bertsub_1_span' + '  --output_dir scire_models/scire-scibert_1_span',

           're_1_span_typemarker_train': commbase_re + '  --model_type bertsub_1_span' + '  --do_train' + '  --output_dir scire_models/scire-scibert_1_span_typemarker'+'  --use_typemarker',
           're_1_span_typemarker_evaluate': commbase_re + '  --model_type bertsub_1_span' + '  --output_dir scire_models/scire-scibert_1_span_typemarker'+'  --use_typemarker',

           're_1_typemarker_train': commbase_re + '  --model_type bertsub_1' + '  --do_train' + '  --output_dir  scire_models/scire-scibert_1-typemarker' + '  --use_typemarker',
           're_1_typemarker_evaluate': commbase_re + '  --model_type bertsub_1' +  '  --output_dir  scire_models/scire-scibert_1-typemarker' + '  --use_typemarker',

           're_levitated_train': commbase_re_levitated + '  --model_type bert' + '  --do_train'+ '  --output_dir  scire_models/scire-scibert-levitatedpair',
           're_levitated_evaluate': commbase_re_levitated + '  --model_type bert' + '  --output_dir  scire_models/scire-scibert-levitatedpair',

           're_bilevitated_train': commbase_re_levitated + '  --model_type bert_blinear' + '  --do_train' + '  --output_dir  scire_models/scire-scibert-bilevitatedpair',
           're_bilevitated_evaluate': commbase_re_levitated + '  --model_type bert_blinear' + '  --output_dir  scire_models/scire-scibert-bilevitatedpair',

           're_bilevitated_1_train': commbase_re_levitated + '  --model_type bert_blinear_1' + '  --do_train' + '  --output_dir  scire_models/scire-scibert-bilevitatedpair-1',
           're_bilevitated_1_evaluate': commbase_re_levitated + '  --model_type bert_blinear_1' + '  --output_dir  scire_models/scire-scibert-bilevitatedpair-1',

           're_bilevitated_typemarker_train': commbase_re_levitated + '  --model_type bert_blinear_1' + '  --do_train' + '  --output_dir  scire_models/scire-scibert-bilevitatedpair-typemarker' + '  --use_typemarker',
           're_bilevitated_typemarker_evaluate': commbase_re_levitated + '  --model_type bert_blinear_1' + '  --output_dir  scire_models/scire-scibert-bilevitatedpair-typemarker' + '  --use_typemarker',

           'ner_train':commbase_ner+'  --model_type bertspanmarker '+'  --do_train' + '  --output_dir sciner_models/sciner-scibert',
           'ner_evaluate': commbase_ner + '  --model_type bertspanmarker' + '  --output_dir sciner_models/sciner-scibert',

           'ner_prompt_train': commbase_ner_prompt + '  --model_type bertspanmarker ' + '  --do_train' + '  --output_dir sciner_models/sciner-scibert-prompt',
           'ner_prompt_evaluate': commbase_ner_prompt + '  --model_type bertspanmarker' + '  --output_dir sciner_models/sciner-scibert-prompt',

           'ner_onlyentity_train':commbase_ner+'  --model_type bertspanmarker_onlyentity '+'  --do_train' + '  --output_dir sciner_models/sciner-onlyentity-scibert',
           'ner_onlymarker_train':commbase_ner + '  --model_type bertspanmarker_onlymarker ' + '   --do_train' + '  --output_dir sciner_models/sciner-onlymarker-scibert',

           'ner_onlyentity_evaluate': commbase_ner + '  --model_type bertspanmarker_onlyentity ' + '  --output_dir sciner_models/sciner-onlyentity-scibert',
           'ner_onlymarker_evaluate': commbase_ner + '  --model_type bertspanmarker_onlymarker '+ '  --output_dir sciner_models/sciner-onlymarker-scibert',

           'ner_bertspanmarkerbi_train': commbase_ner + '  --model_type bertspanmarkerbi ' + '   --do_train' + '  --output_dir sciner_models/sciner-scibert-blinear',
           'ner_bertspanmarkerbi_evaluate': commbase_ner + '  --model_type bertspanmarkerbi ' + '  --output_dir sciner_models/sciner-scibert-blinear',

           'ner_bertspanmarkerbi_spanmarker_train': commbase_ner + '  --model_type bertspanmarkerbi ' + '   --do_train' + '  --output_dir sciner_models/sciner-scibert-blinear'+ '  --use_typemarker',
           'ner_bertspanmarkerbi_spanmarke_evaluate': commbase_ner + '  --model_type bertspanmarkerbi ' + '  --output_dir sciner_models/sciner-scibert-blinear'+ '  --use_typemarker',

           'ner_roberta_train': commbase_robert_ner + '  --model_type robertaspanmarker ' + '  --do_train' + '  --output_dir sciner_models/sciner-sciroberta',
           'ner_roberta_evaluate': commbase_robert_ner + '  --model_type robertaspanmarker' + '  --output_dir sciner_models/sciner-sciroberta',

           'ner_roberta_prompt_train': commbase_ner_prompt + '  --model_type robertaspanmarker ' + '  --do_train' + '  --output_dir sciner_models/sciner-sciroberta-prompt',
           'ner_roberta_prompt_evaluate': commbase_ner_prompt + '  --model_type robertaspanmarker' + '  --output_dir sciner_models/sciner-sciroberta-prompt',


           }
    def __init__(self):
        super().__init__()
        self.title = "界面"
        self.left = 200
        self.top = 200
        self.width = 1450
        self.height = 896
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        palette = QPalette()
        #设置界面背景
        palette.setColor(QPalette.Background, QColor("#fcfcfc"))
        #palette.setBrush(QPalette.Background, QBrush(QPixmap('C:/Users/gao28/Pictures/src=http___pic1.win4000.com_wallpaper_6_54508ba326282.jpg&refer=http___pic1.win4000.webp')))
        self.setPalette(palette),

        self.label = QLabel(self)
        self.label.setText("       绘制训练过程相关曲线")
        self.label.setFixedSize(553, 553)
        self.label.move(896, 0)
        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:30px;font-weight:bold;font-family:JetBrains Mono;"
                                 "border-left: 15px solid #F2F2F2;}"
                                 )


        #定义文本框
        self.text_edit = QTextEdit(self)
        self.text_edit.setStyleSheet("border: 15px solid #F2F2F2;")

        self.text_edit.setReadOnly(True)
        self.text_edit.setFixedSize(1450, 342)
        self.text_edit.move(0, 553)
        self.text_edit.setFontFamily('JetBrains Mono')
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.text_edit.setFont(font)

        # 定义文本框
        self.text_edit_input = QTextEdit(self)
        self.text_edit_input.setStyleSheet("border: 15px solid #F2F2F2; border-bottom: none;border-right: none;")
        self.text_edit_input.setFixedSize(897, 150)
        self.text_edit_input.move(0, 403)
        self.text_edit_input.setFontFamily('JetBrains Mono')
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.text_edit_input.setFont(font)




        self.test_btn = QPushButton(self)
        self.test_btn.setText("测试")
        self.test_btn.move(600, 170)

        self.test_btn.clicked.connect(self.test)

        self.train_btn = QPushButton(self)
        self.train_btn.setText("开始")
        self.train_btn.move(600, 210)

        self.train_btn.clicked.connect(self.train)

        self.draw_btn = QPushButton(self)
        self.draw_btn.setText("展示")
        self.draw_btn.move(600, 250)
        self.draw_btn.clicked.connect(self.draw)

        self.eva_btn = QPushButton(self)
        self.eva_btn.setText("评估")
        self.eva_btn.move(600, 290)

        self.eva_btn.clicked.connect(self.evaluate)

        self.rbtrain = QRadioButton('train', self)
        self.rbevaluate = QRadioButton('evaluate', self)
        self.info = 0
        self.rbtrain.move(600, 100)
        self.rbevaluate.move(600, 130)
        self.rbgroup = QButtonGroup(self)
        self.rbgroup.addButton(self.rbtrain, 1)
        self.rbgroup.addButton(self.rbevaluate, 2)
        self.rbgroup.buttonClicked.connect(self.rbClicked)


        self.combo_box_1 = QComboBox(self)  # 创建第一级 QComboBox 对象
        self.combo_box_1.setFixedSize(300,25)
        self.combo_box_1.move(100,120)

        self.combo_box_2 = QComboBox(self)  # 创建第二级 QComboBox 对象
        self.combo_box_2.setFixedSize(300, 25)
        self.combo_box_2.move(100,170)

        self.combo_box_3 = QComboBox(self)
        self.combo_box_3.setFixedSize(300, 25)
        self.combo_box_3.move(100, 220)

        self.combo_box_4 = QComboBox(self)
        self.combo_box_4.setFixedSize(300, 25)
        self.combo_box_4.move(100, 270)



        self.items_1 = ['ner', 're']
        self.items_2 = {
            'ner': ['ner', 'ner_onlyentity', 'ner_onlymarker','ner_roberta','ner_albert','ner_prompt'],
            're': ['re', 're_1','re_1_span' ,'re_1_span_typemarker','re_albert','re_typemarker','re_levitated','re_bilevitated_1','re_bilevitated','re_1_typemarker','re_bilevitated_typemarker',]
        }
        self.items_3 = {
                         'train_ner_onlymarker_loss','train_re_bilevitated_1_loss'  , 'train_re_levitated_f1',
                         'train_re_1_f1','train_re_bilevitated_f1','train_re_levitated_loss',
                         'train_re_1_loss ','train_re_bilevitated_loss ','train_re_loss','train_ner_onlyentity_f1',
                        'train_re_bilevitated_typemarker_f1','train_re_typermarker_f1',
                        'train_ner_onlyentity_loss','train_re_1_typermarker_loss','train_re_bilevitated_typemarker_loss',
                        'train_re_typermarker_loss','train_ner_onlymarker_f1', 'train_re_bilevitated_1_f1','train_re_f1'
                        'train_re_albert_f1','train_re_albert_loss'}
        self.items_4 = {
'scire-scibert_1','scire-scibert',
'scire-scibert_1_span_typemarker' ,
 'scire-scibert-typemarker',
'scire-scibert_1-typemarker',
'scire-scibert-span',
'scire-scibert_1_span',
            'sciner-scibert',
            'sciner-scibert-prompt',
            'sciner-onlymarker-scibert',
            'sciner-onlyentity-scibert',
            'sciner-sciroberta'
        }

        # 将字符串作为第一项，表示提示信息
        self.combo_box_1.insertItem(0, '请选择阶段')
        # 将当前选中的项设置为提示信息
        self.combo_box_1.setCurrentIndex(0)
        # 将第一级 QComboBox 的选项添加到列表中
        self.combo_box_1.addItems(self.items_1)


        self.combo_box_1.currentIndexChanged.connect(self.combo_box_1_changed)

        self.combo_box_3.addItems(self.items_3)
        # 将字符串作为第一项，表示提示信息
        self.combo_box_3.insertItem(0, '请选择要展示的训练过程文件')
        # 将当前选中的项设置为提示信息
        self.combo_box_3.setCurrentIndex(0)

        self.combo_box_4.addItems(self.items_4)
        # 将字符串作为第一项，表示提示信息
        self.combo_box_4.insertItem(0, '请选择要展示的评估结果文件')
        # 将当前选中的项设置为提示信息
        self.combo_box_4.setCurrentIndex(0)


        self.show()
    def combo_box_1_changed(self,index):
            #当第一级 QComboBox 中的选项被改变时，更新第二级 QComboBox 的选项
            #:param index: 第一级 QComboBox 中的选项索引
        item_text = self.items_1[index-1]
        if item_text in self.items_2:
            self.combo_box_2.clear()  # 清空第二级 QComboBox 的选项
            self.combo_box_2.addItems(self.items_2[item_text])                  #添加新的选项

    def update_output_text(self, data):
        # 更新 QTextEdit 控件中的输出内容
        self.text_edit.moveCursor(QtGui.QTextCursor.End)
        self.text_edit.insertPlainText(data)


    def rbClicked(self):
        sender = self.sender()
        if sender == self.rbgroup:
            if self.rbgroup.checkedId() == 1:
                self.info = 1
            elif self.rbgroup.checkedId() == 2:
                self.info = 2

    def data_read(self,dir_path):
        with open(dir_path, "r") as f:
            raw_data = f.read()
            data = raw_data[1:-1].split(", ")  # [-1:1]是为了去除文件中的前后中括号"[]"

        return np.asfarray(data, float)


    def draw(self):


        if self.combo_box_3.currentIndex() != 0:
            # 弹出文件选择对话框，获取保存文件路径
            save_path, _ = QFileDialog.getSaveFileName(self, '保存文件', self.combo_box_3.currentText(),'Text Files (*.txt)')
            # 如果用户取消了文件选择，则返回
            if not save_path:
                return

            # 获取主机、用户名、密码和文件地址
            host = '10.1.48.237'
            username = 'root'
            password = '1008Zhangys@237'
            filepath = '/home/PL-Marker/scierc/' + self.combo_box_3.currentText() + '.txt'

            # 连接远程主机
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, username=username, password=password)

            # 创建 SFTP 客户端
            sftp = client.open_sftp()

            # 下载文件并保存到本地
            sftp.get(filepath, save_path)

            # 关闭连接
            sftp.close()
            client.close()

            # 绘制f1指数曲线
            y_train_acc = self.data_read(save_path)  # 训练准确率值，即y轴
            x_train_acc = range(len(y_train_acc))  # 训练阶段准确率的数量，即x轴

            # 创建一个Figure对象，并添加一个Axes对象
            fig = Figure()
            ax = fig.add_subplot(111)

            # 去除顶部和右边框框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.set_ylim(0, 2.5)
            ax.set_xlabel('iters')  # x轴标签
            if "f1" in self.combo_box_3.currentText():
                self.index = 'f1'
                ax.set_ylabel('f1')  # y轴标签
            else:
                self.index = 'loss'
                ax.set_ylabel('loss')  # y轴标签

            # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
            # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
            if "ner" in self.combo_box_3.currentText():
                ax.plot(x_train_acc, y_train_acc, linewidth=1, linestyle="solid", label="ner_" + self.index)
            else:
                ax.plot(x_train_acc, y_train_acc, linewidth=1, linestyle="solid", label="re_" + self.index)
            ax.legend()

            ax.set_title(self.combo_box_3.currentText() + '_curve')

            # 创建 FigureCanvas 控件
            self.canvas = FigureCanvas(fig)
            self.canvas.setFixedSize(500, 400)

            # 将 FigureCanvas 中的图像转换为 QPixmap 对象
            pixmap = QPixmap(self.canvas.size())
            self.canvas.render(pixmap)

            # 设置 QLabel 的显示内容为 QPixmap对象
            self.label.setPixmap(pixmap)

            # 读取文件内容
            with open(save_path, 'r') as f:
                file_content = f.read()

            # 输出文件内容,为了调试方便
            print(file_content)

            # 删除文件
            os.remove(save_path)

        if self.combo_box_4.currentIndex() != 0:
            if 'scire' in self.combo_box_4.currentText():
            # 弹出文件选择对话框，获取保存文件路径
                save_path, _ = QFileDialog.getSaveFileName(self, '保存文件', 'pred_results_sentence',
                                                       'JSON Files (*.json)')
            if 'sciner' in self.combo_box_4.currentText():
                save_path, _ = QFileDialog.getSaveFileName(self, '保存文件', 'ent_pred_test',
                                                           'JSON Files (*.json)')
            # 如果用户取消了文件选择，则返回
            if not save_path:
                return

            # 获取主机、用户名、密码和文件地址
            host = '10.1.48.237'
            username = 'root'
            password = '1008Zhangys@237'
            if 'scire' in self.combo_box_4.currentText():
                filepath = '/home/PL-Marker/scire_models/' + self.combo_box_4.currentText() + '/pred_results_sentence'+'.json'
                resultpath = '/home/PL-Marker/scire_models/' + self.combo_box_4.currentText() + '/results'+'.json'
            if 'sciner' in self.combo_box_4.currentText():
                filepath = '/home/PL-Marker/sciner_models/' + self.combo_box_4.currentText() + '/ent_pred_test' + '.json'
                resultpath = '/home/PL-Marker/sciner_models/' + self.combo_box_4.currentText() + '/results' + '.json'
            # 连接远程主机
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(host, username=username, password=password)

            # 创建 SFTP 客户端
            sftp = client.open_sftp()

            # 下载文件并保存到本地
            sftp.get(resultpath, 'D:/毕设/PL-Marker/result.json')
            sftp.get( filepath, save_path)

            self.text_edit.append("begin")
            self.text_edit.repaint()  # 强制更新界面
            # 关闭连接
            sftp.close()
            client.close()
            print("begin")
            text = ""
            count = 0
            if 'scire' in self.combo_box_4.currentText():
                f = open(save_path, "r", encoding='utf-8')


                for l_idx, line in enumerate(f):  #每次读一行sentences，包括很多个句子
                    data = json.loads(line)
                    for i in range(len(data)):
                        sentences = data[str(i)]
                        for j in range(len(sentences)):
                            sentence = sentences[j]
                            if sentence[1] != []:
                                text += '句子: ' + '\n'
                                text += ' '.join(sentence[1][0]) + '\n'+ '\n'
                                text += '当前所选模型预测的句中关系: ' + '\n'+ '\n'
                                for k in range(1, len(sentence[1])):
                                    text += '"' + ' '.join(sentence[1][k][0]) + '" 与' + ' "' + ' '.join(
                                        sentence[1][k][1]) + '" 之间是 "' + sentence[1][k][2] + '"关系。\n'
                                text +=  '\n'
                                #此处加入原始模型的表现
                                f_pl = open('D:/毕设/pl_pred_results_sentence.json', "r", encoding='utf-8')  # 记录原始模型的表现
                                begin_1 = 0
                                for l_idx_2, line_2 in enumerate(f_pl):  # 每次读一行sentences，包括很多个句子
                                    data_2 = json.loads(line_2)
                                    for u in range(len(data_2)):
                                        if u == i:
                                            sentences_2 = data_2[str(u)]
                                            for v in range(len(sentences_2)):
                                                if v == j:
                                                    sentence_2 = sentences_2[v]
                                                    if sentence_2[1] != []:
                                                        text += 'PL-Marker模型预测的句中关系: ' + '\n'+ '\n'
                                                        for y in range(1, len(sentence_2[1])):
                                                            text += '"' + ' '.join(
                                                                sentence_2[1][y][0]) + '" 与' + ' "' + ' '.join(
                                                                sentence_2[1][y][1]) + '" 之间是 "' + sentence_2[1][y][
                                                                        2] + '"关系。\n'
                                                        begin_1 = 1
                                                        text += '\n'
                                            if begin_1 == 1:

                                                break
                                f_pl.close()
                                                #此处应该加入全部实体关系
                                index = 0
                                begin = 0#0代表还没结束
                                # 打开数据集文件
                                f_1 = open('D:/毕设/PL-Marker/scierc/test.json', "r", encoding='utf-8')
                                for l_idx_1, line_1 in enumerate(f_1):  # 每次读一行sentences，包括很多个句子,簇
                                    data_1 = json.loads(line_1)
                                    length = 0
                                    if index == i:#簇相等
                                        sentences_rel = data_1['sentences']
                                        re_rel = data_1['relations']
                                        for x in range(0,j+1):#找句子
                                            if x == j:#如果找到了相同句子
                                                text+='句中实际关系: ' + '\n'+ '\n'
                                                for l in range(len(re_rel[x])):  # 遍历该句子中的关系
                                                    text += '"' + ' '.join(sentences_rel[x][re_rel[x][l][0] - length:re_rel[x][l][1] - length + 1]) + '" 与' + ' "' + ' '.join(sentences_rel[x][re_rel[x][l][2] - length:re_rel[x][l][3] - length + 1]) + '" 之间是 "' + re_rel[x][l][4] + '"关系。\n'
                                                begin = 1#1代表跳出循环
                                            length += len(sentences_rel[x])
                                    index += 1
                                    if begin == 1:

                                        break
                                f_1.close()
                                text += '\n'+ '\n'
                            count += 1
                            if count == 5:  # 处理了一定数量的句子
                                self.text_edit.append(text)
                                self.text_edit.repaint()  # 强制更新界面
                                text = ""  # 清空文本变量以便继续添加更多文本
                                count = 0  # 重置计数器
                f.close()

                if text:  # 如果还有剩余的文本
                    self.text_edit.append(text)
                    self.text_edit.repaint()  # 强制更新界面

                # self.text_edit.setText(text)
                # self.text_edit.repaint()  # 强制更新界面
                print("end")
            if 'sciner' in self.combo_box_4.currentText():
                f = open(save_path, "r", encoding='utf-8')
                count = 0
                for l_idx, line in enumerate(f):  # 每次读一行sentences，包括很多个句子,一个簇
                    data = json.loads(line)
                    sentences = data['sentences']#一个簇n个句子
                    ner = data['predicted_ner']#每个簇的ner结果
                    ner_rel = data['ner']
                    length = 0
                    text = ''
                    begin = 0
                    for i in range(len(sentences)):
                        text += '\n'+'句子：'+'\n'+ '\n'
                        text += ' '.join(sentences[i])+'\n'+'\n'#每个句子第一次处理时，先全部打印出来
                        text += '当前所选模型所预测句中实体： '+'\n'+ '\n'
                        for j in range(len(ner[i])):#遍历一个句子中的多个ner
                            text += '"' + ''.join(sentences[i][ner[i][j][0]-length:ner[i][j][1]+1-length])+'" 的实体类型是："'+''.join(ner[i][j][2])+' "\n'
                        text += '\n'
                        #此处加入PL-Marker模型的结果
                        f_pl = open('D:/毕设/pl_ent_pred_test.json', "r", encoding='utf-8')
                        for l_idx_1, line_1 in enumerate(f_pl):  # 每次读一行sentences，包括很多个句子,一个簇
                            if l_idx_1 == l_idx:
                                data_1 = json.loads(line_1)
                                ner_pl = data_1['predicted_ner']
                                sentences_pl = data_1['sentences']
                                for k in range(len(sentences_pl)):
                                    if k == i:
                                        text += 'PL-Marker模型所预测句中实体： ' + '\n' + '\n'
                                        for m in range(len(ner_pl[k])):  # 遍历一个句子中的多个ner
                                            text += '"' + ''.join(sentences_pl[k][ner_pl[k][m][0] - length:ner_pl[k][m][1] + 1 - length]) + '" 的实体类型是："' + ''.join(
                                                ner_pl[k][m][2]) + ' "\n'
                                        text += '\n'
                                        begin = 1
                                if begin == 1:
                                    break
                        f_pl.close()

                        # 此处应该加入正确的实体类型
                        text += '句中实际实体： ' + '\n'+ '\n'
                        for k in range(len(ner_rel[i])):#遍历一个句子中的多个ner
                            text += '"' + ''.join(sentences[i][ner_rel[i][k][0]-length:ner_rel[i][k][1]+1-length])+'" 的实体类型是："'+''.join(ner_rel[i][k][2])+' "\n'

                        length = length + len(sentences[i])

                    count = count +1
                    if count == 10:
                        self.text_edit.append(text)
                        self.text_edit.repaint()  # 强制更新界面
                        text = ""  # 清空文本变量以便继续添加更多文本
                        count = 0  # 重置计数器
                f.close()
                """f_result = open('D:/毕设/PL-Marker/result.json', "r", encoding='utf-8')
                for l_idx, line in enumerate(f_result):  # 每次读一行sentences，包括很多个句子,一个簇
                    data = json.loads(line)
                    text += '该模型的性能表现为：'+'\n'
                    f1 = data['f1_']
                    presion = data['precision_']
                    recall = data['recall_']
                    text += 'F1: '+str(f1)+'  precision:'+str(presion)+'  recall:'+str(recall)+'\n'
                f_result.close()"""
                if text:  # 如果还有剩余的文本
                    self.text_edit.append(text)
                    self.text_edit.repaint()  #强制更新界面
          # 删除文件
            os.remove(save_path)
            os.remove('D:/毕设/PL-Marker/result.json')

    def test(self):
        # 创建一个 SSHThread 线程实例
        self.thread = SSHThread('10.1.48.237', 22, 'root', '1008Zhangys@237', 'cd /home&&cd ./PL-Marker/scire_models&&ls')
        # 将输出信号连接到槽函数中进行界面更新
        self.thread.output_signal.connect(self.update_output_text)
        # 启动线程
        self.thread.start()
        # 禁用按钮，避免重复连接
        #self.test_btn.setEnabled(False)


    def evaluate(self):
        commbase_ner = 'CUDA_VISIBLE_DEVICES=1  python3  run_ner_test.py   \
            --model_name_or_path  bert_models/scibert_scivocab_uncased  --do_lower_case  \
            --data_dir scierc  \
            --learning_rate 2e-5  --num_train_epochs 50  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
            --max_seq_length 512  --save_steps 263  --max_pair_length 256  --max_mention_ori_length 8    \
            --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
            --fp16  --seed 42  --onedropout  --lminit  \
            --train_file train.json --dev_file dev.json --test_file data.json  \
             --overwrite_output_dir  --output_results'
        commbase_re = 'CUDA_VISIBLE_DEVICES=1  python3  run_re_test.py  \
            --model_name_or_path  bert_models/scibert_scivocab_uncased  --do_lower_case  \
            --data_dir scierc  \
            --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
            --max_seq_length 256  --max_pair_length 16  --save_steps 904  \
            --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
            --fp16   \
            --use_ner_results '

        # 待分词的文本
        text = self.text_edit_input.toPlainText()
        # 加载英文模型
        nlp = spacy.load("en_core_web_sm")
        # 分词
        doc = nlp(text)

        # 将分词结果存入列表
        word_list = [token.text for token in doc]



        # 定义字典变量
        data_dict = {"sentences":None, "ner":None, "relations":None}
        data_dict["ner"] = [[[0,0,"OtherScientificTerm"]]]
        data_dict["relations"] = [[[0,0,0,0,"CONJUNCTION"]]]
        data_dict["sentences"] = [word_list]

        # 指定保存的文件夹
        save_dir = 'D:/毕设/PL-Marker/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 打开并保存字典到json文件
        with open(os.path.join(save_dir, "data.json"), "w") as f:
            json.dump(data_dict, f)

        # 获取主机、用户名、密码和文件地址
        host = '10.1.48.237'
        username = 'root'
        password = '1008Zhangys@237'

        # 连接远程主机
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, username=username, password=password)

        # 创建 SFTP 客户端
        sftp = client.open_sftp()

        # 上传文件
        sftp.put('D:/毕设/PL-Marker/data.json', '/home/PL-Marker/scierc/data.json')


        if  'ner' in str(self.combo_box_2.currentText()):
            comm = commbase_ner + self.model[str(self.combo_box_2.currentText())]
            save_path = 'D:/毕设/PL-Marker/ner_result.json'
        if 're' in str(self.combo_box_2.currentText()):
            comm = commbase_re + self.model[str(self.combo_box_2.currentText())] +'  --test_file  ner_result.json'
            save_path = 'D:/毕设/PL-Marker/re_result.json'

        # 创建一个 SSHThread 线程实例
        self.thread = SSHThread('10.1.48.237', 22, 'root', '1008Zhangys@237',
                                'conda activate teach_duan&&cd /home&&cd ./PL-Marker&&' + comm)
        # 将输出信号连接到槽函数中进行界面更新
        self.thread.output_signal.connect(self.update_output_text)
        # 启动线程
        self.thread.start()

        # 判断线程是否结束
        while self.thread.isRunning():
            # do something while the thread is running
            time.sleep(2)
            pass


        #立即更新界面
        QApplication.processEvents()

        if 'ner' in str(self.combo_box_2.currentText()):
            # 上传文件
            sftp.get('/home/PL-Marker/scierc/ner_result.json', save_path)
        if 're' in str(self.combo_box_2.currentText()):
            # 上传文件
            sftp.get('/home/PL-Marker/scierc/re_result.json', save_path)

        # 关闭SFTP客户端和SSH连接
        sftp.close()
        client.close()

        #展示结果
        ouput = ''
        f = open(save_path, "r", encoding='utf-8')

        if 'ner' in save_path:
            for l_idx, line in enumerate(f):  # 每次读一行sentences，包括很多个句子,一个簇
                data = json.loads(line)
                ner = data['predicted_ner']
                sentences = data['sentences']
                ouput += '句中的实体为：' + '\n'
                for i in range(len(ner[0])):
                    ouput += '"' + ' '.join(sentences[0][ner[0][i][0]:ner[0][i][1] + 1]) + '" 的实体类型是："' + ''.join(
                        ner[0][i][2]) + ' "\n'


        if 're' in save_path:
            for l_idx, line in enumerate(f):  # 每次读一行sentences，包括很多个句子,一个簇
                data = json.loads(line)
                ouput += '句中的关系为: ' + '\n'
                for i in range(len(data)):
                    sentences = data[str(i)]
                    for j in range(len(sentences)):
                        sentence = sentences[j]
                        if sentence[1] != []:

                            for k in range(1, len(sentence[1])):
                                ouput += '"' + ' '.join(sentence[1][k][0]) + '" 与' + ' "' + ' '.join(
                                    sentence[1][k][1]) + '" 之间是 "' + sentence[1][k][2] + '"关系。\n'
                            ouput += '\n'

        self.text_edit.append(ouput)
        self.text_edit.repaint()  # 强制更新界面







    def train(self):
        text = self.combo_box_2.currentText()

        if self.info == 1:
            index = text + '_' + 'train'
            comm = self.comms[index]
        if self.info == 2:
            index = text + '_' + 'evaluate'
            comm = self.comms[index]


        # 创建一个 SSHThread 线程实例
        self.thread = SSHThread('10.1.48.237', 22, 'root', '1008Zhangys@237', 'conda activate teach_duan&&cd /home&&cd ./PL-Marker&&'+comm)
        # 将输出信号连接到槽函数中进行界面更新
        self.thread.output_signal.connect(self.update_output_text)
        # 启动线程
        self.thread.start()
        # 禁用按钮，避免重复连接
        #self.train_btn.setEnabled(False)




    def closeEvent(self, event):
        # 在窗口关闭时终止线程并关闭 SSH 连接
        for thread in QThread.instances():
            if isinstance(thread, SSHThread):
                if thread.isRunning():
                    thread.terminate()
                if thread.ssh.get_transport().is_active():
                    thread.ssh.close()
        event.accept()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())