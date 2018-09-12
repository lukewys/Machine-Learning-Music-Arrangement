print('\n\n\n\n正在导入库文件，请稍等...\n\n\n\n')
from MLBMA_lib import *

time.sleep(1)
print('\n\n库文件导入完成！\n\n')
while (1):
    print('单声部-单音模型-巴赫风格: 请输入"mono_bach"\n')
    print('双声部-多音模型-巴赫风格: 请输入"poly_bach"\n')
    print('双声部-多音模型-混合风格: 请输入"poly_mix"\n')
    out_mode = input("请选择模型：")
    filepath = input("请输入.mid文件名：")
    if out_mode == 'mono_bach':
        generate_mono(filepath, 10)
        break
    if out_mode == 'poly_bach':
        generate_poly(filepath, 'bach')
        break
    if out_mode == 'poly_mix':
        generate_poly(filepath, 'mix')
        break
    else:
        print("输入模型名称错误！")

print("\n\n已输出10首.mid文件！")
