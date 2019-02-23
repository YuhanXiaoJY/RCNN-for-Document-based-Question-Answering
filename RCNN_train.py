# -*- coding: utf-8 -*-
"""
This file contains the code for training and evaluation.
"""
from data_util import *
from BiLSTM import *
import time
import math

def RCNN_train():
    # load pre-trained embedding vector
    print("\n[INFO]: loading embedding...")
    embedding, word2idx = load_embedding(FLAGS.embedding_file)
    e_len = len(embedding)
    print(e_len)


    # load data
    print("\n[INFO]: loading data...")
    train_questions, train_true_answers, train_false_answers, train_question_num = \
        data2index(FLAGS.train_prefile, word2idx,
                            FLAGS.max_sentence_len)

    validation_questions, validation_answers, validation_labels, validation_questions_len = \
        data2index(FLAGS.validation_prefile, word2idx,
                            FLAGS.max_sentence_len)
    print('\n[INFO]: batch info.')
    train_questions_len = len(train_questions)
    batch_size = FLAGS.batch_size
    train_batch_num = math.ceil(train_questions_len / batch_size)
    validation_batch_num = math.ceil(validation_questions_len / batch_size)
    print('train_questions_len: %d' % train_questions_len)
    print('validation_questions_len: %d' % validation_questions_len)
    print('train_batch_size: %d' % batch_size)
    print('train_batch_num: %d' % train_batch_num)
    print('validation_batch_num: %d' % validation_batch_num)

    print('[INFO]: batch info ended.')

    # evaluating
    def evaluate():
        print("evaluating..")
        scores = []

        for val_k in tqdm(range(validation_batch_num)):
            val_start = val_k * batch_size
            val_end = min((val_k + 1) * batch_size, validation_questions_len)
            val_question, val_answer = [], []
            for val_i in range(val_start, val_end):
                val_question.append(validation_questions[val_i])
                val_answer.append(validation_answers[val_i])
            val_question = np.array(val_question)
            val_answer = np.array(val_answer)

            test_feed_dict = {
                lstm.inputTestQuestions: val_question,
                lstm.inputTestAnswers: val_answer,
                lstm.dropout: 1.0
            }
            _, score = sess.run([globalStep, lstm.result], test_feed_dict)
            scores.extend(score)

        # output the scores and use the evaltool to evaluate
        score_file = open(FLAGS.score_file, 'w', encoding='utf-8')
        for score in scores:
            score_file.write('%f\n' % score)
        score_file.close()
        eval_command = 'call '+FLAGS.evaluate_file
        os.system(eval_command)
        with open(FLAGS.result_file, 'r') as f:
            line = f.readline().strip().split('\t')
            MAP = float(line[1])
            line = f.readline().strip().split('\t')
            MRR = float(line[1])
        with open('res/log_last.txt', 'a+') as f:
            f.write('epoch:{}   batch_num:{}    MAP:{}  MRR:{}\n'.format(epoch, k, MAP, MRR))

        print('\n[INFO]: epoch %d trained.' % j)
        return MAP, MRR

    # start training
    print("\ntraining...")
    with tf.Graph().as_default(), tf.device("/gpu:0"):
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0.75
        )
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options,
            log_device_placement=False
        )
        with tf.Session(config=session_conf).as_default() as sess:
            globalStep = tf.Variable(0, name="global_step", trainable=False)
            lstm = BiLSTM(
                FLAGS.batch_size,
                FLAGS.max_sentence_len,
                embedding,
                FLAGS.rnn_size,
                FLAGS.margin
            )
            # define training procedure
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars), FLAGS.max_grad_norm)
            saver = tf.train.Saver()

            # output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
            print("Writing to {}\n".format(out_dir))

            # get summary
            MAP = 0.0
            MRR = 0.0
            tf.summary.scalar('MAP', tf.convert_to_tensor(MAP))
            tf.summary.scalar('MRR', tf.convert_to_tensor(MRR))
            tf.summary.scalar("loss", lstm.loss)
            summary_op = tf.summary.merge_all()

            summary_dir = os.path.join(out_dir, timestamp)
            summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

            # training
            sess.run(tf.global_variables_initializer())
            lr = FLAGS.learning_rate
            for down in range(FLAGS.lr_down_times):
                optimizer = tf.train.GradientDescentOptimizer(lr)
                optimizer.apply_gradients(zip(grads, tvars))
                trainOp = optimizer.apply_gradients(zip(grads, tvars), global_step=globalStep)
                j = 0
                for epoch in range(FLAGS.num_epochs):
                    print('\n[INFO]: start to train epoch %d' % j)

                    for k in tqdm(range(train_batch_num)):
                        start = k*batch_size
                        end = min((k+1) * batch_size, train_questions_len)
                        question, trueAnswer, falseAnswer = [], [], []
                        for i in range(start, end):
                            question.append(train_questions[i])
                            trueAnswer.append(train_true_answers[i])
                            falseAnswer.append(train_false_answers[i])
                        question = np.array(question)
                        trueAnswer = np.array(trueAnswer)
                        falseAnswer = np.array(falseAnswer)

                        feed_dict = {
                            lstm.inputQuestions: question,
                            lstm.inputTrueAnswers: trueAnswer,
                            lstm.inputFalseAnswers: falseAnswer,
                            lstm.dropout: FLAGS.dropout_keep_prob,
                        }
                        _, step, _, _, loss, summary = \
                            sess.run([trainOp, globalStep, lstm.trueCosSim, lstm.falseCosSim, lstm.loss, summary_op],
                                    feed_dict)

                        if step % FLAGS.evaluate_every == 0:
                            print("step:", step, "loss:", loss)
                            # MAP, MRR = evaluate()
                            summary_writer.add_summary(summary, step)

                    MAP, MRR = evaluate()
                    saver.save(sess, FLAGS.save_file)
                    j += 1
                lr *= FLAGS.lr_down_rate
            evaluate()



if __name__ == '__main__':
    RCNN_train()