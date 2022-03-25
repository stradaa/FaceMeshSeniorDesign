# from Code.face_mesh_mediapipe import MediaPipe_Method
# from GCM.geometric_computation import Geometric_Computation
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton, MDRectangleFlatButton
from kivymd.uix.label import MDLabel
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2


class MyApp(MDApp):
    def build(self):
        # themes
        self.theme_cls.primary_palette = 'Green'
        self.theme_cls.primary_hue = '200'

        # main layout components
        self.web_feed = Image()  # Feed
        self.frame1 = Image()  # Frame 1
        self.frame2 = Image()  # Frame 2
        self.label = MDLabel(text="NeuroVA",
                             halign='center',
                             theme_text_color='Custom',
                             text_color=(0 / 255.0, 204 / 255.0, 102 / 255.0, 1),
                             font_style='H2')

        layout = MDBoxLayout(orientation='vertical', padding=10, spacing=15)
        layout.add_widget(self.label)
        layout.add_widget(self.web_feed)

        inner_box = MDBoxLayout(orientation='horizontal', spacing=0)
        inner_box.add_widget(self.frame1)
        inner_box.add_widget(self.frame2)
        layout.add_widget(inner_box)

        inner_buttons = MDBoxLayout(orientation='horizontal', spacing=10)
        self.save_img1_button = MDRaisedButton(text="Capture 1",
                                               pos_hint={'center_x': 0.5, 'center_y': 0.5},
                                               size_hint=(None, None))
        self.save_img1_button.bind(on_press=self.take_picture)
        inner_buttons.add_widget(self.save_img1_button)
        self.save_img2_button = MDRaisedButton(text="Capture 2",
                                               pos_hint={'center_x': 0.5, 'center_y': 0.5},
                                               size_hint=(None, None))
        self.save_img2_button.bind(on_press=self.take_picture2)
        inner_buttons.add_widget(self.save_img2_button)
        layout.add_widget(inner_buttons)

        self.submit_button = MDRectangleFlatButton(text="Submit",
                                                   pos_hint={'center_x': 0.5, 'center_y': 0.5},
                                                   size_hint=(None, None))
        layout.add_widget(self.submit_button)

        # 30 fps capture
        self.capture = cv2.VideoCapture(1)
        Clock.schedule_interval(self.load_video, 1.0 / 30.0)
        return layout

    def load_video(self, *args):
        # read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        self.frame = frame

        # Flip horizontal and convert image to texture
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.web_feed.texture = texture

    def take_picture(self, *args):
        frame = self.frame
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.frame1.texture = texture

        # cv2.imshow("cv2 final image", img)
        # cv2.imwrite(image_name, self.frame)

    def take_picture2(self, *args):
        frame = self.frame
        buffer = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.frame2.texture = texture

        # cv2.imshow("cv2 final image", img)
        # cv2.imwrite(image_name, self.frame)

    def mediapipe(self, *args):
        pass


MyApp().run()
