def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        # assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'cycle_attn_gan':
        from .cycle_attn_gan_model import CycleAttnGANModel
        model = CycleAttnGANModel()
    elif opt.model == 'cycle_attn_gan_sar':
        from .cycle_attn_gan_model_sar import CycleAttnGANModel_sar
        model = CycleAttnGANModel_sar()
    elif opt.model == 'cycle_attn_gan_sar2':
        from .cycle_attn_gan_model_sar2 import CycleAttnGANModel_sar2   #不管用
        model = CycleAttnGANModel_sar2()
    elif opt.model == 'pix2pix':
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'pix2pix2':
        from .pix2pix_model2 import Pix2PixModel2
        model = Pix2PixModel2()
    elif opt.model == 'pix2pix3':
        from .pix2pix_model3 import Pix2PixModel3
        model = Pix2PixModel3()
    elif opt.model == 'pix2pix_attn':
        from .pix2pix_attn_model import Pix2Pix_attn_Model
        model = Pix2Pix_attn_Model()

    elif opt.model == 'pix2pix_attn2':
        from .pix2pix_attn_model2 import Pix2Pix_attn_Model2
        model = Pix2Pix_attn_Model2()

    elif opt.model == 'pix2pix_attn3':
        from .pix2pix_attn_model3 import Pix2Pix_attn_Model3
        model = Pix2Pix_attn_Model3()

    elif opt.model == 'pix2pix_attn4':
        from .pix2pix_attn_model4 import Pix2Pix_attn_Model4
        model = Pix2Pix_attn_Model4()

    elif opt.model == 'pix2pix_attn5':
        from .pix2pix_attn_model5 import Pix2Pix_attn_Model5
        model = Pix2Pix_attn_Model5()

    elif opt.model == 'pix2pix_attn6':
        from .pix2pix_attn_model6 import Pix2Pix_attn_Model6
        model = Pix2Pix_attn_Model6()
    elif opt.model == 'pix2pix_attn7':
        from .pix2pix_attn_model7 import Pix2Pix_attn_Model7
        model = Pix2Pix_attn_Model7()
    elif opt.model == 'pix2pix_attn8':
        from .pix2pix_attn_model8 import Pix2Pix_attn_Model8
        model = Pix2Pix_attn_Model8()

    elif opt.model == 'Saroptgan':
        from .Saroptgan import Saroptgan
        model = Saroptgan()

    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
