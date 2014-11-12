# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import logging
import math

from PIL import Image

log_file = '%s.log' % __name__
logging.basicConfig(filename=log_file, level=logging.DEBUG)


def apply_grayscale(img, xy, grayscale, fname='grayscale.pgm'):
    """
    Aplica os tons de cinza na imagem
    :param img: the image
    :param xy: dimensions
    :param grayscale: tones
    :param fname: filename
    """
    start, delta = 0, 0
    x, y = xy
    new_img = Image.new('L', (x, y))
    delta = 256 / grayscale
    grayscale = delta

    while start < 256:
        for i in range(x):
            for j in range(y):
                if img.getpixel((i, j)) >= start and img.getpixel((i, j)) < grayscale:
                    if start == 0:
                        new_img.putpixel((i, j), start)
                    else:
                        new_img.putpixel((i, j), grayscale - 1)
        start = grayscale
        grayscale += delta

    new_img.save(fname)

    return new_img, fname


def apply_mask(xy, img, new_img, mask):
    """
    Aplica uma mascara na matriz de uma imagem
    :param xy:
    :param img:
    :param new_img:
    :param mask:
    :return:
    """
    x, y = xy
    for i in range(1, x - 1):
        for j in range(1, y - 1):
            pixel = mask[0][0] * img.getpixel((i - 1, j - 1))
            pixel += mask[0][1] * img.getpixel((i - 1, j))
            pixel += mask[0][2] * img.getpixel((i - 1, j + 1))
            pixel += mask[1][0] * img.getpixel((i, j - 1))
            pixel += mask[1][1] * img.getpixel((i, j))
            pixel += mask[1][2] * img.getpixel((i, j + 1))
            pixel += mask[2][0] * img.getpixel((i + 1, j - 1))
            pixel += mask[2][1] * img.getpixel((i + 1, j))
            pixel += mask[2][2] * img.getpixel((i + 1, j + 1))

            new_img.putpixel((i, j), pixel)


def apply_sobel_mask(xy, img, new_img, sobel_left, sobel_right):
    """
    Aplica a mascara de sobel
    :param xy:
    :param img:
    :param new_img:
    :param sobel_left:
    :param sobel_right:
    :return:
    """
    x, y = xy
    for i in range(1, x - 1):
        for j in range(1, y - 1):
            pixel_left = sobel_left[0][0] * img.getpixel((i - 1, j - 1))
            pixel_left += sobel_left[0][1] * img.getpixel((i - 1, j))
            pixel_left += sobel_left[0][2] * img.getpixel((i - 1, j + 1))
            pixel_left += sobel_left[1][0] * img.getpixel((i, j - 1))
            pixel_left += sobel_left[1][1] * img.getpixel((i, j))
            pixel_left += sobel_left[1][2] * img.getpixel((i, j + 1))
            pixel_left += sobel_left[2][0] * img.getpixel((i + 1, j - 1))
            pixel_left += sobel_left[2][1] * img.getpixel((i + 1, j))
            pixel_left += sobel_left[2][2] * img.getpixel((i + 1, j + 1))

            pixel_right = sobel_right[0][0] * img.getpixel((i - 1, j - 1))
            pixel_right += sobel_right[0][1] * img.getpixel((i - 1, j))
            pixel_right += sobel_right[0][2] * img.getpixel((i - 1, j + 1))
            pixel_right += sobel_right[1][0] * img.getpixel((i, j - 1))
            pixel_right += sobel_right[1][1] * img.getpixel((i, j))
            pixel_right += sobel_right[1][2] * img.getpixel((i, j + 1))
            pixel_right += sobel_right[2][0] * img.getpixel((i + 1, j - 1))
            pixel_right += sobel_right[2][1] * img.getpixel((i + 1, j))
            pixel_right += sobel_right[2][2] * img.getpixel((i + 1, j + 1))

            new_img.putpixel((i, j), math.sqrt(math.pow(pixel_left, 2) + math.pow(pixel_right, 2)))


def apply_smoothing(img, xy, fname='smoothing.pgm'):
    """
    Aplica o filtro de suavizacao
    :param img:
    :param xy: dimensions
    :param fname: file name
    """
    new_img = Image.new('L', xy)
    smooth_matrix = [[1 / 9.0 for i in range(3)] for i in range(3)]

    apply_mask(xy, img, new_img, smooth_matrix)

    new_img.save(fname)

    return new_img, fname


def apply_better_details(img, xy, fname='better_details.pgm'):
    """
    Aplica filtro de realce
    :param img:
    :param xy: dimensions
    :param fname: file name
    """
    new_img = Image.new('L', xy)
    matrix = [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]]
    apply_mask(xy, img, new_img, matrix)
    new_img.save(fname)
    return new_img, fname


def apply_border_north(img, xy, fname='borders-north.pgm'):
    """
    :param img:
    :param xy:
    :param fname:
    :return:
    """
    new_img = Image.new('L', xy)

    # north
    matrix = [[1.0, 1.0, 1.0], [1.0, -2.0, 1.0], [-1.0, -1.0, -1.0]]
    apply_mask(xy, img, new_img, matrix)
    new_img.save(fname)
    return new_img, fname


def apply_border_south(img, xy, fname='borders-south.pgm'):
    """

    :param img:
    :param xy:
    :param fname:
    :return:
    """
    new_img = Image.new('L', xy)

    # south
    matrix = [[-1.0, -1.0, -1.0], [1.0, -2.0, 1.0], [1.0, 1.0, 1.0]]
    apply_mask(xy, img, new_img, matrix)
    new_img.save(fname)
    return new_img, fname


def apply_border_east(img, xy, fname='borders-east.pgm'):
    """
    :param img:
    :param xy:
    :param fname:
    :return:
    """
    new_img = Image.new('L', xy)

    matrix = [[-1.0, 1.0, 1.0], [-1.0, -2.0, 1.0], [-1.0, 1.0, 1.0]]
    apply_mask(xy, img, new_img, matrix)
    new_img.save(fname)
    return new_img, fname


def apply_border_west(img, xy, fname='borders-west.pgm'):
    """
    :param img:
    :param xy:
    :param fname:
    :return:
    """
    new_img = Image.new('L', xy)

    matrix = [[1.0, 1.0, -1.0], [1.0, -2.0, -1.0], [1.0, 1.0, -1.0]]
    apply_mask(xy, img, new_img, matrix)
    new_img.save(fname)
    return new_img, fname


def apply_laplace(img, xy, fname='laplace.pgm'):
    """
    :param img: the image
    :param xy: the dimensions
    :param fname: the filename
    :return: new image
    """
    new_img = Image.new('L', xy)
    matrix = [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    apply_mask(xy, img, new_img, matrix)
    new_img.save(fname)
    return new_img, fname


def apply_sobel(img, xy, fname='sobel.pgm'):
    """
    :param img: the image
    :param xy: the dimensions
    :param fname: the filename
    :return: new image
    """
    new_img = Image.new('L', xy)
    sobel_right = [[-1.0, 2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    sobel_left = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    apply_sobel_mask(xy, img, new_img, sobel_left, sobel_right)
    new_img.save(fname)
    return new_img, fname


if __name__ == '__main__':
    print '-==== Aplicacao de filtros em imagens ====-'
    print 'v1.0'
    optout = True
    image = None
    new_image = None
    filename = None
    try:
        while optout is True:
            if not image:
                image_filename = raw_input('Digite o nome do arquivo PGM: ')
                try:
                    logging.debug('Abrindo imagem %s...' % image_filename)
                    image = Image.open(image_filename)
                    logging.debug('Imagem %s carregada com sucesso.' % image_filename)
                    print 'Imagem \'%s\' carregada com sucesso!' % image_filename
                    dimensions = (256, 256)
                except IOError:
                    print 'Imagem \'%s\' nao encontrada!' % image_filename
                    image = None

            if image:
                option = int(raw_input('\nEscolha o que deseja fazer:\n'
                                       '1 - Aplicar uma quantidade de tons de cinza\n'
                                       '2 - Aplicar filtro Suavizacao\n'
                                       '3 - Aplicar filtro Realce de Detalhes\n'
                                       '4 - Aplicar filtro de Laplace\n'
                                       '5 - Aplicar filtro de Sobel\n'
                                       '6 - Aplicar filtro nas bordas\n'
                                       '7 - Visualizar histograma da ultima imagem gerada\n'
                                       '8 - Visualizar histograma da imagem original\n'
                                       '9 - Sair\n\n'
                                       'Digite a opcao desejada: '))

                if option == 1:
                    tones = int(raw_input('Quantidade de tons de cinza a ser aplicada: '))
                    print 'Processando...'
                    new_image, filename = apply_grayscale(image, dimensions, tones)
                    print 'Uma nova imagem foi gerada e salva com sucesso!'
                    preview_image = np.asarray(new_image)
                    plt.imshow(preview_image, cmap=cm.Greys_r)
                    plt.show()
                elif option == 2:
                    print 'Processando...'
                    logging.debug('Aplicando filtro de suavizacao...')
                    new_image, filename = apply_smoothing(image, dimensions)
                    logging.debug('Filtro de suavizacao aplicado com sucesso. Imagem: %s' % filename)
                    print 'Uma nova imagem foi gerada e salva com sucesso!'
                    preview_image = np.asarray(new_image)
                    plt.imshow(preview_image, cmap=cm.Greys_r)
                    plt.show()
                elif option == 3:
                    print 'Processando...'
                    logging.debug('Aplicando filtro de realce de detalhes...')
                    new_image, filename = apply_better_details(image, dimensions)
                    print 'Uma nova imagem foi gerada e salva com sucesso!'
                    logging.debug('Filtro de realce de detalhes aplicado com sucesso. Imagem: %s' % filename)
                    preview_image = np.asarray(new_image)
                    plt.imshow(preview_image, cmap=cm.Greys_r)
                    plt.show()
                elif option == 4:
                    print 'Processando...'
                    logging.debug('Aplicando filtro de Laplace...')
                    new_image, filename = apply_laplace(image, dimensions)
                    print 'Uma nova imagem foi gerada e salva com sucesso!'
                    logging.debug('Filtro de laplace aplicado com sucesso. Imagem: %s' % filename)
                    preview_image = np.asarray(new_image)
                    plt.imshow(preview_image, cmap=cm.Greys_r)
                    plt.show()
                elif option == 5:
                    print 'Processando...'
                    logging.debug('Aplicando filtro de Sobel...')
                    new_image, filename = apply_sobel(image, dimensions)
                    print 'Uma nova imagem foi gerada e salva com sucesso!'
                    logging.debug('Filtro de Sobel aplicado com sucesso. Imagem: %s' % filename)
                    preview_image = np.asarray(new_image)
                    plt.imshow(preview_image, cmap=cm.Greys_r)
                    plt.show()
                elif option == 6:
                    border_option = int(raw_input('\nEscolha a borda:\n'
                                                  '1 - Norte\n'
                                                  '2 - Sul\n'
                                                  '3 - Leste\n'
                                                  '4 - Oeste\n\n'
                                                  'Digite a opcao de borda desejada: '))
                    proceed = False
                    if border_option == 1:
                        proceed = True
                        print 'Processando...'
                        logging.debug('Aplicando filtro na borda norte...')
                        new_image, filename = apply_border_north(image, dimensions)
                        logging.debug('Filtro na borda norte aplicado com sucesso. Imagem: %s' % filename)
                        print 'Uma nova imagem foi gerada e salva com sucesso!'
                    elif border_option == 2:
                        proceed = True
                        print 'Processando...'
                        logging.debug('Aplicando filtro na borda sul...')
                        new_image, filename = apply_border_south(image, dimensions)
                        logging.debug('Filtro na borda sul aplicado com sucesso. Imagem: %s' % filename)
                        print 'Uma nova imagem foi gerada e salva com sucesso!'
                    elif border_option == 3:
                        proceed = True
                        print 'Processando...'
                        logging.debug('Aplicando filtro na borda leste...')
                        new_image, filename = apply_border_east(image, dimensions)
                        logging.debug('Filtro na borda leste aplicado com sucesso. Imagem: %s' % filename)
                        print 'Uma nova imagem foi gerada e salva com sucesso!'
                    elif border_option == 4:
                        proceed = True
                        print 'Processando...'
                        logging.debug('Aplicando filtro na borda oeste...')
                        new_image, filename = apply_border_west(image, dimensions)
                        logging.debug('Filtro na borda oeste aplicado com sucesso. Imagem: %s' % filename)
                        print 'Uma nova imagem foi gerada e salva com sucesso!'
                    else:
                        proceed = False
                        print 'Opcao de borda inexistente!'

                    if proceed:
                        preview_image = np.asarray(new_image)
                        plt.imshow(preview_image, cmap=cm.Greys_r)
                        plt.show()
                elif option == 7:
                    if new_image:
                        logging.debug('Criando histograma para a ultima imagem gerada...')
                        preview_image = np.asarray(new_image)
                        plt.hist(preview_image.flatten(), bins=100, fc='black', alpha=0.75)
                        plt.title('Histograma da Imagem: %s' % filename or 'Not defined.')
                        plt.xlabel('Tons de Cinza')
                        plt.ylabel('Pixels')
                        plt.show()
                        logging.debug('Histograma foi gerado com sucesso!')
                    else:
                        print 'Nenhuma imagem foi gerada ate o momento!'
                elif option == 8:
                    if image:
                        logging.debug('Criando histograma para a ultima imagem gerada...')
                        preview_image = np.asarray(image)
                        plt.hist(preview_image.flatten(), bins=100, fc='black', alpha=0.75)
                        plt.title('Histograma da Imagem: %s' % image_filename)
                        plt.xlabel('Tons de Cinza')
                        plt.ylabel('Pixels')
                        plt.show()
                        logging.debug('Histograma da imagem original foi gerado com sucesso!')
                    else:
                        print 'Nenhuma imagem foi carregada ate o momento!'
                elif option == 9:
                    print 'See you soon!'
                    optout = False
                else:
                    print '%d Nao eh uma opcao valida!' % option
    except KeyboardInterrupt:
        print '\n\nSee you soon!'
