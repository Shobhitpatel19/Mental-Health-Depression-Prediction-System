import './App.css';
import HomeContentWidget from './components/HomeContentWidget';
import {Route, Routes } from 'react-router-dom';
import NavBar from './components/NavBar';
import AboutSection from './components/AboutSection';
import ContactUsSection from './components/ContactUsSection';

function App() {
  return (
    <div>
      <NavBar />
      
        <Routes>
          <Route path="/" element={<HomeContentWidget />} />
          <Route path="/about" element={<AboutSection />} />
          <Route path='/contactUs' element={<ContactUsSection />} />
        </Routes>
     
    </div>

  );
}

export default App;
