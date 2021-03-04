import tensorflow as tf
import numpy as np
import cv2
import pathlib
import os
import pandas as pd

def loadvideo(filename: str):
    """Loads a video from a file.
    Args:
        filename (str): filename of video
    Returns:
        A np.ndarray with dimensions (frames, height, width). The
        values will be uint8's ranging from 0 to 255.
    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_width, frame_height), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v[count] = frame

    return v

def normalize_negative_one(video):
    normalized_input = (video - np.amin(video)) / (np.amax(video) - np.amin(video))
    return 2*normalized_input - 1


class KvaeDataLoader():
    def __init__(self, root, d_type, length, image_shape = (64,64), test=False):
        assert d_type in ['box', 'box_gravity', 'polygon', 'pong'], "d_type {0} not supported".format(d_type)
        self.image_shape = image_shape
        self.length = length
        if test:
            f = '{0}_test.npz'.format(d_type)
        else:
            f = '{0}.npz'.format(d_type)
        
        npzfile = np.load(os.path.join(root, f))
        self.videos = npzfile['images'].astype(np.float32)
        data = tf.data.Dataset.from_generator(
            self.generator(),
            tuple([tf.float32, tf.bool]),
            tuple([(self.length, *self.image_shape),(self.length)]))
        self.data = data
    
    def _read_video(self, video):
        video = np.asarray([cv2.resize(video[i,...], dsize=self.image_shape,interpolation=cv2.INTER_CUBIC) for i in range(video.shape[0])])
        video = normalize_negative_one(video)[:self.length,...]
        mask = np.zeros(video.shape[0], dtype='bool')
        return video, mask

    def generator(self):
        def gen():
            for v in self.videos:
                video, mask = self._read_video(v)
                yield tuple([video, mask])
        return gen

class TensorflowDatasetLoader():
    def __init__(self, 
                 root=None, 
                 image_shape = (64,64), 
                 split = 'train', 
                 length = 20, 
                 period=2,
                 max_length=250, 
                 clips=1,
                 pad=None):
        if root is None:
            root = '/data/Niklas/EchoNet-Dynamics'
        
        if clips != 1:
            raise NotImplementedError("Clips other than 1 is not implemented")
        if pad is not None:
            raise NotImplementedError("Pads is not implemented")
        self.image_shape = image_shape
        self.folder = pathlib.Path(root)
        self.split = split
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        
        self.idxs = []
        df = pd.read_csv(self.folder / "FileList.csv")
        for index, row in df.iterrows():
            fileMode = row['Split'].lower()
            fileName = row['FileName'] + '.avi'
            if split in ["all", fileMode] and os.path.exists(self.folder / "Videos" / fileName):
                if fileName in short:
                    continue
                self.idxs.append(fileName)

        data = tf.data.Dataset.from_generator(
            self.generator(),
            tuple([tf.float32, tf.bool]),
            tuple([(self.length, *self.image_shape),(self.length)]))
        self.data = data
    
    def _read_video(self, idx):
        video = os.path.join(self.folder, "Videos", idx)
        video = loadvideo(video).astype(np.float32)          
        f, h, w = video.shape
        mask = np.zeros(f, dtype='bool')
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((length * self.period - f, h, w), video.dtype)), axis=0)
            mask = np.concatenate((mask, np.ones((length * self.period - f), "bool")), axis=0)
            f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            if f - (length) * self.period > 0:
                start = np.random.choice(f - (length) * self.period, self.clips)
            else:
                start = [0]

        video = tuple(video[s + self.period * np.arange(length), :, :] for s in start)
        mask = tuple(mask[s + self.period * np.arange(length)] for s in start)
        if self.clips == 1:
            video = video[0]
            mask = mask[0]
        else:
            video = np.stack(video)
            mask = np.stack(mask)

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            l, h, w = video.shape
            temp = np.zeros((l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]

        if self.image_shape is not None:
            video = np.asarray([cv2.resize(video[i,...], dsize=self.image_shape,interpolation=cv2.INTER_CUBIC) for i in range(video.shape[0])])
        video = normalize_negative_one(video)
        return video, mask
    def generator(self):
        def gen():
            for idx in self.idxs:
                video, mask = self._read_video(idx)
                if np.any(mask == True):
                    print(idx)           
                yield tuple([video, mask])
        return gen


short = ['0X106766224781FAE2.avi',
'0X10D734CBEB6ECB81.avi',
'0X1185DA5AB0D9BE6A.avi',
'0X11C5F414FDFFA6FE.avi',
'0X1205884E675E58B9.avi',
'0X12430512E2BBCD55.avi',
'0X12628F033A56EB6F.avi',
'0X12C17194EECA74B4.avi',
'0X1369E49954F0EAEC.avi',
'0X13D6BC74D088CACA.avi',
'0X150D2E5A2E09ADC.avi',
'0X160DBDFED541D2D9.avi',
'0X1631C52C25D15DEE.avi',
'0X163D02E41A3A93D.avi',
'0X166B0D5A9407635F.avi',
'0X16FE5AFA8320ACD3.avi',
'0X171C87FA9E52210F.avi',
'0X174A9943AB689618.avi',
'0X178DF5359B89AD06.avi',
'0X17D6DA5B33D4069C.avi',
'0X1847E925CBCEB6CE.avi',
'0X197DF6A1F73E7DFC.avi',
'0X19AC1E6345765C03.avi',
'0X19B9109694DE642.avi',
'0X1C8054277A139381.avi',
'0X1CAD9DE1F782AF55.avi',
'0X1CEBFD3654DFE9D3.avi',
'0X1D026A00CF69D6B.avi',
'0X1D22C277EE981F30.avi',
'0X1D5C7F82BC641385.avi',
'0X1E3950AF8323C7C3.avi',
'0X1E41619BEA0A4042.avi',
'0X1EA7A5960D8A5576.avi',
'0X1F7426166F480035.avi',
'0X2040D80BE038FF72.avi',
'0X2067611AEB6EAC1C.avi',
'0X20E888D4879745B8.avi',
'0X21A4FF7D4640E7FC.avi',
'0X21B497B1AE639C14.avi',
'0X21DB11780A6A35EE.avi',
'0X21E16CBE16849CF8.avi',
'0X24E187AB50EC7783.avi',
'0X251F4272F05D83ED.avi',
'0X2641B2DB74DE3FF1.avi',
'0X2795D1D6CC01F594.avi',
'0X27ED61D9D1D7C33.avi',
'0X286661146EB02EE4.avi',
'0X2878D9240A9B5CC4.avi',
'0X288EB79F76CDA89F.avi',
'0X28923B334611CFC1.avi',
'0X29670BE6399C86BC.avi',
'0X29C17E1745706401.avi',
'0X2A55659AE64722AA.avi',
'0X2B8D66D3AF8E8B84.avi',
'0X2BE0908F65A5A564.avi',
'0X2C693B4B714F5F77.avi',
'0X2EE799C4FDC049DB.avi',
'0X2F1EFBACA8FA6CB5.avi',
'0X2F7EF90430E9370C.avi',
'0X302567FE70174845.avi',
'0X3047DAECF3BF4F35.avi',
'0X3064D2C12E6718AD.avi',
'0X3182DE90EDD4C06A.avi',
'0X33BF00D55528C824.avi',
'0X3475D2DF28CA8EFA.avi',
'0X352D7150FCBFA7C1.avi',
'0X354B37A25C64276F.avi',
'0X3572B6AE5C661529.avi',
'0X3590CC9889D55995.avi',
'0X3693781992586497.avi',
'0X385071ADD66846AD.avi',
'0X3991463690B05802.avi',
'0X399873AFA4B674D6.avi',
'0X39FCEE4F2B02701B.avi',
'0X3A04D1EE256B301E.avi',
'0X3A6E1BAA9065D40.avi',
'0X3AB11C73780B865.avi',
'0X3B31E6C62C7C6591.avi',
'0X3BCAEEF6C5EE3565.avi',
'0X3BD7B111C743A3E8.avi',
'0X3BFAE73D52A18389.avi',
'0X3C05A6B4BBA623EB.avi',
'0X3C1F4A30EA25362E.avi',
'0X3C29707A418FF73A.avi',
'0X3C2DCCC974C5380D.avi',
'0X3CF7C368106B59F2.avi',
'0X3D8A8864DB8B12E.avi',
'0X3DBF6E850E8E7FA.avi',
'0X3E6AE8BEB9B6DBD4.avi',
'0X3F1D93F58B1FD1FD.avi',
'0X3F9A02817B34BB11.avi',
'0X4013D6C4137D4297.avi',
'0X40D38248ACAECAFF.avi',
'0X40F475A3CA97F09B.avi',
'0X411DD3F7DC7803DE.avi',
'0X4176EA075A0EBF26.avi',
'0X422C6837BE6E90FF.avi',
'0X42D0EE9B93BD8553.avi',
'0X42FC131122748839.avi',
'0X440008B474A3319D.avi',
'0X453EC33BC6E96460.avi',
'0X4550086B27E8492F.avi',
'0X464D690506890716.avi',
'0X46B1CCE5B354B752.avi',
'0X470C3A06C890B2B3.avi',
'0X4726A11E93B9DE90.avi',
'0X47FC02860FC36935.avi',
'0X4851ABCA6F1C46AD.avi',
'0X49422B590F116D76.avi',
'0X49AA5AE78CD7E71F.avi',
'0X4A792B82556BDA0A.avi',
'0X4A7A4FAAE293863A.avi',
'0X4CEDA292EED01C94.avi',
'0X4D0A1712CDE848D1.avi',
'0X4D160D96C3F7E1FC.avi',
'0X4D386EB1CFBBF2E0.avi',
'0X4DD8FD6672F5D940.avi',
'0X4E69D799006FE37A.avi',
'0X4E75F3A54E92EE54.avi',
'0X4EA078CC4E65B6A3.avi',
'0X4F55DBA9FA2C8A16.avi',
'0X4FEC9915044DD151.avi',
'0X51327FBFD448A550.avi',
'0X51DEA404C6DAE247.avi',
'0X51F276725BEE4BF9.avi',
'0X5238017D9647223F.avi',
'0X52619FC2739EB1F1.avi',
'0X52CC0BAB001C7C06.avi',
'0X52FEE70BDB811E9B.avi',
'0X5333CEAC223AB6D0.avi',
'0X5389ED952B455A33.avi',
'0X53C185263415AA4F.avi',
'0X540CEAFF8921F79A.avi',
'0X541C0693140E6AD0.avi',
'0X5482AE6485A16527.avi',
'0X54F3797A02026F4.avi',
'0X552F36A6E92D7FE8.avi',
'0X558FCFDEDFCAFCD5.avi',
'0X561EA6D2BA7109CC.avi',
'0X563EE078BB4DBE66.avi',
'0X577895B05C084B1C.avi',
'0X57AF4D24B154C573.avi',
'0X57E623AB38FB2122.avi',
'0X5816941DEDFD5EF8.avi',
'0X596D82CEC244AD7B.avi',
'0X5A31A52541B52293.avi',
'0X5A3DD4EB5CEDEBC8.avi',
'0X5B0FF6323B769A36.avi',
'0X5B10B8EF336047D5.avi',
'0X5B4D0D38F575846D.avi',
'0X5C433F9E3BA1E2DF.avi',
'0X5E212B2953D6257B.avi',
'0X5E5C9EB6814282D7.avi',
'0X5FDC113292CB6B3C.avi',
'0X5FE6439A0CCEF482.avi',
'0X6047BEFB56F3FBA8.avi',
'0X60616EC56172D3AE.avi',
'0X6062D547329B34C8.avi',
'0X606543C2636A46F9.avi',
'0X608CC80E40F3C7C7.avi',
'0X6093A86B26D1D702.avi',
'0X60A115AC08C5AA98.avi',
'0X60A3DA455577B5DD.avi',
'0X61B6E8FB5F3B7F28.avi',
'0X6294FD3F7487AA1D.avi',
'0X62B3D1284E90B92E.avi',
'0X62EA10A73F0557B9.avi',
'0X63399D9E4FC71C87.avi',
'0X635245BE8BDD5C3B.avi',
'0X6381C1737EEC6177.avi',
'0X63822A2B901BCE0B.avi',
'0X64EE9FFA5DA69058.avi',
'0X6517121A1CC175DC.avi',
'0X655399113DAA4ECC.avi',
'0X65E2A0A6C4CDBC1.avi',
'0X65E605F203321860.avi',
'0X665B15BB5ABDCCFF.avi',
'0X6668ACFCA23901D4.avi',
'0X667955B6FC13C36.avi',
'0X66B7661D9ADBD284.avi',
'0X66CF7DDCF1B8920D.avi',
'0X67ABFBDF2329C556.avi',
'0X67EEAB34427A2162.avi',
'0X683BC0A201466229.avi',
'0X6986F2D866C7AF73.avi',
'0X698851733B8DB622.avi',
'0X69C480D704708196.avi',
'0X6A46321A41C98FC7.avi',
'0X6A517CFB3CC37E24.avi',
'0X6A8AA36725720569.avi',
'0X6AA45821A706A75B.avi',
'0X6AFD6474F2A92942.avi',
'0X6B9AD5750DEB23A6.avi',
'0X6BB792FB95F3E857.avi',
'0X6C64E0A64888EDBD.avi',
'0X6CBEFE70A7A9BBF9.avi',
'0X6D9E5AB91D96CCE1.avi',
'0X6F4F950A5396343A.avi',
'0X6FAD9DF07824297B.avi',
'0X6FC8A2C44917FE58.avi',
'0X7083497ECE270656.avi',
'0X71F7F25CC893F72.avi',
'0X72DB58A40280753B.avi',
'0X72E41E49681C6CEB.avi',
'0X730D323F56D3D95B.avi',
'0X73E3BB1E0A5DAADB.avi',
'0X74084E3472EDBC5B.avi',
'0X74B6576AF56AECD4.avi',
'0X753AA26EA352BBB.avi',
'0X76265CF61DC60700.avi',
'0X764C542E3E41E4D2.avi',
'0X777110E46D3D382.avi',
'0X779CF7E0C461311.avi',
'0X790DD9CAD22463CA.avi',
'0X796FE72A47DAC27A.avi',
'0X7AB103757B8047FC.avi',
'0X7AFF5B2DEEE839B9.avi',
'0X7C1078ADB86F63B2.avi',
'0X7C3724DE3C2BBEA6.avi',
'0X7C65DA147A4EA7E1.avi',
'0X7CA133BB031E0FFD.avi',
'0X7CAE371CDC897CE6.avi',
'0X7CE0D85A16659110.avi',
'0X7D687BF5AC8C33FF.avi',
'0X7D78994E4A3FCC6D.avi',
'0X7E0ACE153EE2260F.avi',
'0X7E15A2E78B07ADE2.avi',
'0X7E22E4FE22BE624B.avi',
'0X7ED1E92FFFD1AF5C.avi',
'0X7ED2FA1C9CE297BF.avi',
'0X848378E4CE0BD88.avi',
'0X86C408DFC3FF659.avi',
'0X8D4753B7DADB6A6.avi',
'0X973E4A9DAADDF9F.avi',
'0X983B0EA3147F367.avi',
'0X98AE6BD83F02077.avi',
'0X9E181B942CACADF.avi',
'0XAF14E70264D4B68.avi',
'0XAF4A04FFB4BBBC8.avi',
'0XC649E4EF24DCAB7.avi',
'0XCBC6A16E66E2941.avi',
'0XD0BE4647858725A.avi',
'0XD51E16EDF0F45F2.avi',
'0XEE7E2A3E141E060.avi',
'0X2DC68261CBCC04AE.avi',
'0X1498806542CEE367.avi',
'0X1ACB73BE8C1F2C0C.avi',
'0X1C8C0CE25970C40.avi',
'0X1CF4B07994B62DBB.avi',
'0X2CA927607A7AB65A.avi',
'0X2CAA96B0D5EB2D62.avi',
'0X2E8342A7771D0391.avi',
'0X333A6598916DD19.avi',
'0X33C3D95C20A0D931.avi',
'0X41705247E0540BD7.avi',
'0X41AC5C5FC2E3352A.avi',
'0X4225D03BC0BFD7BA.avi',
'0X451AAC4A666A91BF.avi',
'0X47DBEA2F11240016.avi',
'0X4BFD23141A6091C5.avi',
'0X4C81F018685C5333.avi',
'0X4E9F08061D109568.avi',
'0X55D97A446839D999.avi',
'0X574915FE8F47DA9D.avi',
'0X57614735C29E7850.avi',
'0X5AB4C1E0766C1301.avi',
'0X5BA6C5333369EBF9.avi',
'0X5C41AF3F25757D5B.avi',
'0X5DAB9FDE7CC700DE.avi',
'0X5F40FC2C2367EA92.avi',
'0X656705047098A5FC.avi',
'0X685A4B045C95B72A.avi',
'0X686F756A1F380DAC.avi',
'0X69447E46FEDD2A3F.avi',
'0X705789F6BD6B3A46.avi',
'0X7415C8B1BD8C89F0.avi',
'0X753D4BE480775FE7.avi',
'0X756A5DE88BADCAD5.avi',
'0X763034E9D22CB7E1.avi',
'0X77A6EEED554F8360.avi',
'0X7DA74EAC9DFC2D5B.avi',
'0X7E51B1F0EBC1BD6C.avi',
'0X7F058A3503090EC8.avi',
'0XB11B61E6027ABCE.avi',
'0XE9217FEF2BCB4B1.avi',
'0X13E488CF5C7C934C.avi',
'0X185672E7C8FF1212.avi',
'0X1D8D28801D701B6.avi',
'0X1ED182EA95B3349E.avi',
'0X2812459832695FC3.avi',
'0X2C0B21D4520985EB.avi',
'0X2F82938DF2A52427.avi',
'0X30A9476068D69B16.avi',
'0X362AF0DB0ECE89E9.avi',
'0X36BD2518C9D15985.avi',
'0X39CA8CC96A5D5E8B.avi',
'0X3E73331470DA4030.avi',
'0X41F8C8319BD36BF2.avi',
'0X47EDDE2F8004F754.avi',
'0X51DF5916C8A5E638.avi',
'0X588C869314F5D067.avi',
'0X5CFB2DE05E52418E.avi',
'0X64D81B504D7FF03C.avi',
'0X65A9AD514CAB60F6.avi',
'0X6752E91D593ABCAE.avi',
'0X695AF94E37EA61C5.avi',
'0X6D8C2E4FF14F1B01.avi',
'0X6E379A4025B0D43C.avi',
'0X6FE03765A01F42BB.avi',
'0X740E3E79F3D8FD83.avi',
'0X77A68EB0325B1DFC.avi',
'0X7889C9C82EAA329B.avi',
'0X7E1703D6C05CEE3F.avi',
'0XC31605A9D22702E.avi',
'0XC89A68BA0AA0A11.avi']